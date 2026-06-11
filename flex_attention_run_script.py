# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
# TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_probe \
# python3 flex_attention2.py
# #强制重新编译
# 
import torch
import argparse
import functools
import os
import time
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    BlockMask,
)

from sparse_masks import get_sparse_config, list_sparse_configs

# Reorder module for block-level KV reordering
try:
    from torch_npu._inductor.kernel.flex_attention_reorder import (
        reorder_flex_forward,
        compute_block_hit_rate,
        unpermute_output,
        make_reordered_score_mod,
        compute_and_set_pending_perm,
        REORDER_REGISTRY,
    )
    _HAS_REORDER = True
except ImportError:
    _HAS_REORDER = False
    REORDER_REGISTRY = {}


def _prepend_ld_library_paths(*paths):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(path) for path in paths if Path(path).is_dir()]
    for path in existing.split(os.pathsep):
        if path and path not in parts:
            parts.append(path)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


torch.set_float32_matmul_precision("high")
MSTX_DOMAIN = "flex_attention2"
_TORCH_NPU = None
_TORCH_NPU_IMPORT_ERROR = None
_NPU_INDUCTOR_READY = False


def import_torch_npu(required=False):
    global _TORCH_NPU, _TORCH_NPU_IMPORT_ERROR
    if _TORCH_NPU is not None:
        return _TORCH_NPU
    if _TORCH_NPU_IMPORT_ERROR is not None:
        if required:
            raise RuntimeError("torch_npu is required for NPU execution") from _TORCH_NPU_IMPORT_ERROR
        return None
    try:
        import torch_npu as module
    except Exception as exc:
        _TORCH_NPU_IMPORT_ERROR = exc
        if required:
            raise RuntimeError("torch_npu is required for NPU execution") from exc
        return None

    _TORCH_NPU = module
    _prepend_ld_library_paths(
        Path(torch.__file__).resolve().parent / "lib",
        Path(module.__file__).resolve().parent / "lib",
    )
    return module


def device_is_npu(device):
    return str(device).startswith("npu")


def npu_is_available():
    module = import_torch_npu(required=False)
    if module is None or not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def ensure_npu_inductor():
    global _NPU_INDUCTOR_READY
    if _NPU_INDUCTOR_READY:
        return
    import_torch_npu(required=True)
    try:
        import torch_npu._inductor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import torch_npu._inductor. Check that the NPU runtime is "
            "visible to this Python process and that torch/torch_npu versions match."
        ) from exc
    _NPU_INDUCTOR_READY = True
#---------- 原始 flexattention 相关函数 ----------
def create_attention(
    score_mod,
    block_mask,
    enable_gqa=False,
    block_m=64,
    block_n=64,
    kernel_options_extra=None,
):
    kernel_options = {"BLOCK_M": block_m, "BLOCK_N": block_n}
    if kernel_options_extra:
        kernel_options.update(kernel_options_extra)
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )
def identity(score, batch, head, token_q, token_kv):
    return score
def causal_mask(batch, head, token_q, token_kv):
    return token_q >= token_kv
#---------- 小算子拼接 Attention ----------
def build_dense_mask(mask_mod, seq_len, device, dtype):
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
    mask_bool = mask_mod(0, 0, row_indices, col_indices)  # [S, S]
    zero = torch.zeros((), dtype=dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    return torch.where(mask_bool, zero, neg_inf).unsqueeze(0).unsqueeze(0)


def manualattention(q, k, v, mask_mod, dense_mask=None, scale=None, debug=False):
    """
    用基础 PyTorch 算子实现带 mask 的 Attention。
    maskmod 与原 flexattention 中的定义一致：
    maskmod(batch, head, tokenq, tokenkv) -> bool（True 表示保留，False 表示屏蔽）
    """
    if debug:
        print("running 手动attn")
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if dense_mask is None:
        dense_mask = build_dense_mask(mask_mod, S, q.device, q.dtype)

    # 2. QK^T / sqrt(d)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

    # 3. 加上 mask
    attn_scores = attn_scores + dense_mask  # broadcast

    # 4. Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # 5. 加权输出
    output = torch.matmul(attn_weights, v)
    return output
#---------- 配置参数 ----------
B, H, S, D = 4, 8, 2048, 128
SHAPE_SUITES = {
    "single": [(B, H, S, D)],
    "small": [
        (1, 2, 128, 64),
        (1, 4, 256, 64),
        (2, 4, 512, 64),
    ],
    "smoke": [
        (1, 4, 512, 64),
        (2, 8, 1024, 64),
        (B, H, S, D),
    ],
}
test_device = "auto"
test_dtypes = [torch.bfloat16]
test_shapes = SHAPE_SUITES["small"]
test_score_mask_mod_map = {identity: causal_mask}   # 键为 scoremod，值为 maskmod


def parse_shape_spec(spec):
    cleaned = spec.lower().replace("x", ",").replace(":", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"Shape must have four dims B,H,S,D or BxHxSxD, got: {spec}")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Shape dims must be positive, got: {spec}")
    return shape


def selected_shapes(args):
    if getattr(args, "selected_shapes", None) is not None:
        return list(args.selected_shapes)
    if getattr(args, "shape", None):
        shapes = [parse_shape_spec(spec) for spec in args.shape]
    else:
        shapes = list(SHAPE_SUITES[args.shape_suite])
        if args.shape_suite == "single":
            shapes = [(args.batch, args.heads, args.seq_len, args.head_dim)]
    if args.max_shapes is not None:
        shapes = shapes[:args.max_shapes]
    return shapes


def args_for_shape(args, shape):
    shape_args = SimpleNamespace(**vars(args))
    shape_args.batch, shape_args.heads, shape_args.seq_len, shape_args.head_dim = shape
    return shape_args


def dtype_from_name(name):
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return dtype_map[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def dtype_name(dtype):
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_seed(seed):
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)


def set_device(device):
    if device_is_npu(device):
        torch.npu.set_device(device)


def sync_device(device):
    if device_is_npu(device):
        torch.npu.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def release_device_memory(device):
    try:
        sync_device(device)
    except Exception:
        pass
    if device_is_npu(device) and hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def resolve_device(args):
    requested = str(args.device)
    if requested == "auto":
        args.device = "npu" if npu_is_available() else "cpu"
        print(f"resolved --device auto -> {args.device}")
        return args.device

    if device_is_npu(requested):
        import_torch_npu(required=True)
        if not npu_is_available():
            raise RuntimeError(
                f"Requested --device {requested}, but torch.npu.is_available() is False. "
                "If npu-smi works on the host, check that this Python process/container "
                "has the Ascend device and runtime mounted."
            )

    return requested


def make_default_args(**overrides):
    defaults = dict(
        mode="benchmark",
        target="both",
        batch=B,
        heads=H,
        seq_len=S,
        head_dim=D,
        dtype="bfloat16",
        device=test_device,
        warmup=10,
        repeat=10,
        seed=0,
        block_m=64,
        block_n=64,
        manual_mask="precompute",
        dynamic_compile=False,
        allow_npu_dynamic_compile=False,
        enable_gqa=False,
        mstx=False,
        compare=True,
        rtol=None,
        atol=None,
        topk=10,
        msprof_output=None,
        msprof_aic_metrics="PipeUtilization",
        msprof_option=[],
        prescale_qk=False,
        num_warps=None,
        num_stages=None,
        shape=[],
        shape_suite="single",
        max_shapes=None,
        continue_on_shape_error=True,
        selected_shapes=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_inputs(args):
    set_seed(args.seed)
    set_device(args.device)
    dtype = dtype_from_name(args.dtype)
    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, dtype=dtype, device=args.device)
    k = torch.randn(shape, dtype=dtype, device=args.device)
    v = torch.randn(shape, dtype=dtype, device=args.device)
    return q, k, v


def make_flex_runner(q, k, v, score_mod, mask_mod, args, block_mask=None, optimizations=None):
    use_npu = device_is_npu(args.device)
    if use_npu:
        ensure_npu_inductor()

    block_mask_device = "cpu" if use_npu else args.device
    if block_mask is None:
        block_mask = create_block_mask(
            mask_mod,
            1,
            1,
            args.seq_len,
            args.seq_len,
            device=block_mask_device,
        ).to(args.device)
    kernel_options_extra = {}
    if optimizations is not None:
        kernel_options_extra.update(optimizations)
    elif mask_mod is causal_mask:
        kernel_options_extra["ROWS_GUARANTEED_SAFE"] = True
        kernel_options_extra["BLOCKS_ARE_CONTIGUOUS"] = True
    if args.prescale_qk:
        kernel_options_extra["PRESCALE_QK"] = True
    if args.num_warps is not None:
        kernel_options_extra["num_warps"] = args.num_warps
    if args.num_stages is not None:
        kernel_options_extra["num_stages"] = args.num_stages

    sdpa_fn = create_attention(
        score_mod,
        block_mask=block_mask,
        enable_gqa=args.enable_gqa,
        block_m=args.block_m,
        block_n=args.block_n,
        kernel_options_extra=kernel_options_extra,
    )

    if not use_npu:
        if args.dynamic_compile:
            print("Ignoring --dynamic-compile for non-NPU flex attention; using eager execution.")

        def run():
            return sdpa_fn(q, k, v)

        return run

    dynamic_compile = args.dynamic_compile
    if dynamic_compile and not args.allow_npu_dynamic_compile:
        print(
            "Ignoring --dynamic-compile for NPU flex attention because this path is "
            "unstable with torch_npu Inductor; use --allow-npu-dynamic-compile to force it."
        )
        dynamic_compile = False

    compiled_sdpa = torch.compile(
        sdpa_fn,
        backend="inductor",
        dynamic=dynamic_compile,
    )

    def run():
        #print(f"Running flex")
        return compiled_sdpa(q, k, v)

    return run


def make_manual_runner(q, k, v, mask_mod, args):
    dense_mask = None
    if args.manual_mask == "precompute":
        dense_mask = build_dense_mask(mask_mod, args.seq_len, args.device, q.dtype)

    def run():
        return manualattention(q, k, v, mask_mod, dense_mask=dense_mask)

    return run


def mstx_start(message, enabled):
    if not enabled:
        return None
    return import_torch_npu(required=True).npu.mstx.range_start(message, domain=MSTX_DOMAIN)


def mstx_end(range_id, enabled):
    if enabled and range_id is not None:
        import_torch_npu(required=True).npu.mstx.range_end(range_id, domain=MSTX_DOMAIN)


def mstx_mark(message, enabled):
    if enabled:
        import_torch_npu(required=True).npu.mstx.mark(message, domain=MSTX_DOMAIN)


def time_runner(label, runner, args):
    last_output = None
    with torch.no_grad():
        for _ in range(args.warmup):
            last_output = runner()
        sync_device(args.device)

        safe_label = label.lower().replace(" ", "_")
        repeat_name = (
            f"profile_repeat_loop target={args.target} label={safe_label} "
            f"warmup={args.warmup} repeat={args.repeat}"
        )
        mstx_mark(f"profile_repeat_start target={args.target} label={safe_label}", args.mstx)
        range_id = mstx_start(repeat_name, args.mstx)
        sync_device(args.device)
        start_time = time.perf_counter()
        for _ in range(args.repeat):
            last_output = runner()
        sync_device(args.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mstx_end(range_id, args.mstx)
        mstx_mark(f"profile_repeat_end target={args.target} label={safe_label}", args.mstx)

    avg_ms = elapsed_ms / args.repeat
    RED = "\033[31m"
    RESET = "\033[0m"

    print(
        f"B:{args.batch} H:{args.heads} S:{args.seq_len} D:{args.head_dim} "
        f"| {label} avg: {RED}{avg_ms:.3f} ms{RESET} "
        f"(warmup={args.warmup}, repeat={args.repeat})"
    )
    return last_output, avg_ms


def tolerance_for(dtype, rtol, atol):
    if rtol is not None and atol is not None:
        return rtol, atol
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-3, 1e-5


def detailed_compare(output_flex, output_manual, rtol, atol, topk=10):
    eps = 1e-6
    flex_f32 = output_flex.float()
    manual_f32 = output_manual.float()
    absdiff = (flex_f32 - manual_f32).abs()


    max_abs_t = absdiff.max()          # tensor scalar
    max_abs = max_abs_t.item()        # 你的 max_abs（item）
    # 找到 max_abs 对应的下标 (b,h,s,d)
    flat_idx = absdiff.view(-1).argmax()  # tensor scalar
    b, h, s, d = torch.unravel_index(flat_idx, absdiff.shape)

    mean_abs = absdiff.mean().item()
    median_abs = absdiff.median().item()

    # 取对应下标处的 flex/manual 值
    flex_val = output_flex[b, h, s, d]
    manual_val = output_manual[b, h, s, d]

    denom = (flex_val + manual_val) / 2
    pct = max_abs / (denom.item() + eps) * 100
    max_rel_pct = abs(pct)
    #The max diff ratio : (a-b)/[(a+b)*2]
    # print(f"Max abs diff at (b={b}, h={h}, s={s}, d={d}): "
    #     f"flex={flex_val:.6g}, manual={manual_val:.6g}, "
    #     f"abs_diff={max_abs:.6g}, rel_diff={max_rel_pct:.2f}%"
    # )

    threshold = atol + rtol * manual_f32.abs()
    fail_mask = absdiff > threshold
    num_fail = int(fail_mask.sum().item())
    total = manual_f32.numel()
    fail_ratio = num_fail / total

    any_nan_flex = torch.isnan(flex_f32).any().item()
    any_inf_flex = torch.isinf(flex_f32).any().item()
    any_nan_manual = torch.isnan(manual_f32).any().item()
    any_inf_manual = torch.isinf(manual_f32).any().item()

    print("-------- 差异统计 --------")
    print(f"Flex dtype={output_flex.dtype}, Manual dtype={output_manual.dtype}")
    print(
        "nan/inf: "
        f"Flex(nan={any_nan_flex}, inf={any_inf_flex}), "
        f"Manual(nan={any_nan_manual}, inf={any_inf_manual})"
    )
    print(
        f"max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel_pct:.2f}%, "
        f"mean_abs_diff={mean_abs:.6g}, median_abs_diff={median_abs:.6g}"
    )
    print(f"fail_ratio={fail_ratio * 100:.4f}%  (num_fail={num_fail}/{total})")
    print("--------------------------")

    if topk is not None and topk > 0 and (num_fail > 0 or max_rel_pct > 0.05):
        flat_abs = absdiff.reshape(-1)
        k = min(topk, flat_abs.numel())
        vals, idxs = torch.topk(flat_abs, k)
        batch, heads, seq_len, head_dim = output_manual.shape
        
        if max_rel_pct > 3:
            print(f"The max diff ratio (a-b)/[(a+b)*2] bigger than 3% ")
            print(f"Top-{k} absolute differences:")
            for i in range(k):
                flat_idx = idxs[i].item()
                d_idx = flat_idx % head_dim
                tmp = flat_idx // head_dim
                s_idx = tmp % seq_len
                tmp = tmp // seq_len
                h_idx = tmp % heads
                b_idx = tmp // heads

                fv = flex_f32[b_idx, h_idx, s_idx, d_idx].item()
                mv = manual_f32[b_idx, h_idx, s_idx, d_idx].item()
                av = vals[i].item()
                rv = av / max(abs(fv), abs(mv), eps)
                print(
                    f"top{i}: absdiff={av:.6g}, rel={rv * 100:.6g}% "
                    f"@ (b={b_idx}, h={h_idx}, s={s_idx}, d={d_idx}) "
                    f"Flex={fv:.6g}, Manual={mv:.6g}"
                )

    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel_pct,
        "fail_ratio": fail_ratio,
        "num_fail": num_fail,
        "any_nan_flex": any_nan_flex,
        "any_inf_flex": any_inf_flex,
        "any_nan_manual": any_nan_manual,
        "any_inf_manual": any_inf_manual,
    }


def run_benchmark(args, score_mod=identity, mask_mod=causal_mask, optimizations=None, extra_args=None):
    if extra_args is None:
        extra_args = {}
    resolve_device(args)
    q, k, v = make_inputs(args)
    outputs = {}
    timings = {}

    if args.target in ("both", "flex"):
        # Handle pre-built block mask for patterns like random_block_sparse
        block_mask_override = None
        if extra_args.get("build_block_mask_fn") and mask_mod is None:
            fn = extra_args["build_block_mask_fn"]
            kv_num, kv_idx, simple_mask = fn(args.seq_len)
            kv_num_bh = kv_num.unsqueeze(0).unsqueeze(0)
            kv_idx_bh = kv_idx.unsqueeze(0).unsqueeze(0)
            block_mask_override = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_bh,
                kv_indices=kv_idx_bh,
                full_kv_num_blocks=torch.zeros_like(kv_num_bh),
                full_kv_indices=torch.zeros_like(kv_idx_bh),
                BLOCK_SIZE=(128, 128),
                mask_mod=simple_mask,
            ).to(args.device)
            # Use the simple mask_mod for the flex runner
            flex_runner = make_flex_runner(q, k, v, score_mod, simple_mask, args,
                                           block_mask=block_mask_override,
                                           optimizations=optimizations)
        else:
            flex_runner = make_flex_runner(q, k, v, score_mod, mask_mod, args,
                                           optimizations=optimizations)
        outputs["flex"], timings["flex"] = time_runner("Flex Attention", flex_runner, args)

    if args.target in ("both", "manual") and mask_mod is not None:
        manual_runner = make_manual_runner(q, k, v, mask_mod, args)
        outputs["manual"], timings["manual"] = time_runner("Manual Attention", manual_runner, args)

    # ── Reorder variant: kernel-internal via set_pending_perm ──
    reorder_hit_rate = None
    if getattr(args, "enable_block_reorder", False) and _HAS_REORDER and device_is_npu(args.device):
        block_mask_device = "cpu" if device_is_npu(args.device) else args.device

        # Use pre-built block mask if available, otherwise create from mask_mod
        build_fn = extra_args.get("build_block_mask_fn")
        if build_fn and args.sparse_config:
            kv_num, kv_idx, _ = build_fn(args.seq_len)
            kv_num_bh = kv_num.unsqueeze(0).unsqueeze(0)
            kv_idx_bh = kv_idx.unsqueeze(0).unsqueeze(0)
            full_kv_num = torch.zeros_like(kv_num_bh)
            full_kv_idx = torch.zeros_like(kv_idx_bh)
            bm = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_bh, kv_indices=kv_idx_bh,
                full_kv_num_blocks=full_kv_num, full_kv_indices=full_kv_idx,
                BLOCK_SIZE=(128, 128), mask_mod=mask_mod,
            ).to(args.device)
        else:
            bm = create_block_mask(
                mask_mod, 1, 1, args.seq_len, args.seq_len,
                device=block_mask_device,
            ).to(args.device)

        baseline_hit = compute_block_hit_rate(
            bm.kv_indices, bm.kv_num_blocks,
            bm.full_kv_indices, bm.full_kv_num_blocks,
        )

        try:
            perm = compute_and_set_pending_perm(
                bm.kv_num_blocks, bm.kv_indices,
                bm.full_kv_num_blocks, bm.full_kv_indices,
                mode=args.block_reorder_mode,
                wave_size=args.wave_size,
                verbose=True,
            )
        except Exception as e:
            print(f"[reorder] Error computing permutation: {e}")
            perm = None

        if perm is not None:
            # Kernel-internal reorder: perm is set, kernel will use it.
            # No need to reorder Q, modify block_mask, or unpermute output.
            reorder_runner = make_flex_runner(
                q, k, v, score_mod, mask_mod, args,
                block_mask=bm,
                optimizations=optimizations,
            )

            outputs["flex_reorder"], timings["flex_reorder"] = time_runner(
                f"Flex+{args.block_reorder_mode}", reorder_runner, args)

            # Compute reordered hit rate for display
            reordered_hit = compute_block_hit_rate(
                bm.kv_indices, bm.kv_num_blocks,
                bm.full_kv_indices, bm.full_kv_num_blocks,
            )
            reorder_hit_rate = (baseline_hit, reordered_hit)
        else:
            reorder_hit_rate = (baseline_hit, baseline_hit)
            print("[reorder] Identity permutation — reorder skipped")

    close = None
    stats = None
    if args.target == "both" and args.compare:
        dtype = dtype_from_name(args.dtype)
        rtol, atol = tolerance_for(dtype, args.rtol, args.atol)
        print(f"rtol={rtol}, atol={atol}")
        close = torch.allclose(
            outputs["flex"].float(),
            outputs["manual"].float(),
            rtol=rtol,
            atol=atol,
        )
        stats = detailed_compare(
            outputs["flex"],
            outputs["manual"],
            rtol=rtol,
            atol=atol,
            topk=args.topk,
        )
        print("✅ 测试通过（allclose=True）" if close else "❌ 测试失败（allclose=False）")

        if reorder_hit_rate is not None:
            print(f"  Hit rate: {reorder_hit_rate[0]:.4f} → {reorder_hit_rate[1]:.4f} "
                  f"(delta: {reorder_hit_rate[1] - reorder_hit_rate[0]:+.4f})")

        # ── Reorder vs manual comparison (not done above) ──
        if "flex_reorder" in outputs and "manual" in outputs and args.compare:
            print("\n--- flex_reorder vs manual ---")
            close_reorder = torch.allclose(
                outputs["flex_reorder"].float(),
                outputs["manual"].float(),
                rtol=rtol,
                atol=atol,
            )
            reorder_stats = detailed_compare(
                outputs["flex_reorder"],
                outputs["manual"],
                rtol=rtol,
                atol=atol,
                topk=args.topk,
            )

            # Extra NaN analysis for reorder output
            o_reorder = outputs["flex_reorder"]
            o_manual = outputs["manual"]
            if torch.isnan(o_reorder).any():
                nan_mask = torch.isnan(o_reorder)
                nan_count = nan_mask.sum().item()
                total = o_reorder.numel()
                print(f"  NaN count: {nan_count}/{total} ({100.0*nan_count/total:.2f}%)")
                # Check if NaN is per-sequence-position
                nan_per_pos = nan_mask.any(dim=-1).any(dim=1).float().mean(dim=0)  # [S]
                nan_seq_positions = torch.where(nan_per_pos > 0)[0]
                print(f"  Number of sequence positions with any NaN: {len(nan_seq_positions)}/{o_reorder.shape[2]}")
                # Check if NaN correlates with reorder
                if reorder_hit_rate:
                    bsl, reord = reorder_hit_rate
                    print(f"  (note: hit rate is {bsl:.4f} -> {reord:.4f}, hit rate=0 may indicate empty block rows)")

            print("✅ reorder 测试通过（allclose=True）" if close_reorder else "❌ reorder 测试失败（allclose=False）")

    return {
        "outputs": outputs,
        "timings": timings,
        "close": close,
        "stats": stats,
        "reorder_hit_rate": reorder_hit_rate,
    }


def run_shape_sweep(args, score_mod=identity, mask_mod=causal_mask, optimizations=None, extra_args=None):
    if extra_args is None:
        extra_args = {}
    shapes = selected_shapes(args)
    if len(shapes) == 1:
        return run_benchmark(args_for_shape(args, shapes[0]), score_mod=score_mod, mask_mod=mask_mod,
                             optimizations=optimizations, extra_args=extra_args)

    shape_results = []
    print(f"running shape sweep: {len(shapes)} shapes")
    for index, shape in enumerate(shapes, start=1):
        shape_args = args_for_shape(args, shape)
        print(
            f"\n=== shape {index}/{len(shapes)}: "
            f"B:{shape[0]} H:{shape[1]} S:{shape[2]} D:{shape[3]} ==="
        )
        try:
            result = run_benchmark(shape_args, score_mod=score_mod, mask_mod=mask_mod,
                                    optimizations=optimizations, extra_args=extra_args)
            shape_results.append((shape, result, None))
        except RuntimeError as exc:
            message = str(exc).splitlines()[0]
            print(f"shape failed: {type(exc).__name__}: {message}")
            shape_results.append((shape, None, exc))
            if not args.continue_on_shape_error:
                raise
        finally:
            release_device_memory(shape_args.device)

    print("\n-------- shape sweep summary --------")
    failed = False
    for shape, result, error in shape_results:
        label = f"B:{shape[0]} H:{shape[1]} S:{shape[2]} D:{shape[3]}"
        if error is not None:
            failed = True
            print(f"{label} | error={type(error).__name__}")
        else:
            close = result["close"]
            if close is False:
                failed = True
            timings = ", ".join(
                f"{name}={value:.3f}ms" for name, value in result["timings"].items()
            )
            print(f"{label} | close={close} | {timings}")
    print("-------------------------------------")

    return {
        "shape_results": shape_results,
        "close": not failed,
    }


def profile_target(args):
    resolve_device(args)
    print(
        f"profile target={args.target}, pid={os.getpid()}, "
        f"manual_mask={args.manual_mask}, dynamic_compile={args.dynamic_compile}, "
        f"mstx={args.mstx}"
    )
    args.compare = False
    run_benchmark(args)


def target_argv_for_msprof(args, target):
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "profile-target",
        "--target",
        target,
        "--batch",
        str(args.batch),
        "--heads",
        str(args.heads),
        "--seq-len",
        str(args.seq_len),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--block-m",
        str(args.block_m),
        "--block-n",
        str(args.block_n),
        "--manual-mask",
        args.manual_mask,
        "--mstx",
    ]
    if args.dynamic_compile:
        argv.append("--dynamic-compile")
    if args.allow_npu_dynamic_compile:
        argv.append("--allow-npu-dynamic-compile")
    if args.enable_gqa:
        argv.append("--enable-gqa")
    if args.prescale_qk:
        argv.append("--prescale-qk")
    if args.num_warps is not None:
        argv.extend(["--num-warps", str(args.num_warps)])
    if args.num_stages is not None:
        argv.extend(["--num-stages", str(args.num_stages)])
    return argv


def run_msprof(args):
    msprof_bin = shutil.which("msprof")
    if msprof_bin is None:
        raise RuntimeError("msprof not found on PATH")

    targets = ["flex", "manual"] if args.target == "both" else [args.target]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.msprof_output or f"msprof_out/{timestamp}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        out_dir = output_root / target
        out_dir.mkdir(parents=True, exist_ok=True)
        app = shlex.join(target_argv_for_msprof(args, target))
        cmd = [
            msprof_bin,
            f"--output={out_dir}",
            f"--application={app}",
            "--msproftx=on",
            f"--aic-metrics={args.msprof_aic_metrics}",
            "--ai-core=on",
            "--task-time=on",
            "--runtime-api=on",
            f"--mstx-domain-include={MSTX_DOMAIN}",
        ]
        cmd.extend(args.msprof_option)
        print("Running:", shlex.join(cmd))
        subprocess.run(cmd, check=True)
        print(f"msprof output for {target}: {out_dir}")


SWEEP_SHAPES = [
    "1,4,512,64",
    "2,4,512,64",
    "2,8,1024,64",
    "4,8,2048,128",
]

# Configs that are safe for small/medium shapes (no hardcoded seq_len assumptions)
_SMALL_SPARSE_CONFIGS = [
    "causal",
    "sliding_window_64",
    "sliding_window_128",
    "global_local",
    "nested",
    "prefix_lm",
    "dilated_window",
    "strided",
]


def _resolve_sparse_configs(args):
    """Parse --sparse-config into a list of config names.

    Supported values:
      - None / "all" : all configs from list_sparse_configs()
      - "small"       : 8 configs safe for small shapes
      - "causal,sliding_window_64,..." : comma-separated list
    """
    val = args.sparse_config
    if val is None or val == "all":
        return list_sparse_configs()
    if val == "small":
        return list(_SMALL_SPARSE_CONFIGS)
    return [c.strip() for c in val.split(",") if c.strip()]


def run_sweep_mode(args):
    """Run sparse configs × shapes in subprocess isolation.

    Each (config, shape) pair runs as a fresh subprocess to avoid
    torch.compile / Inductor state pollution between different sparse patterns.

    Use --sparse-config to control which configs (default: all).
      --sparse-config small   → 8 safe configs for small shapes
      --sparse-config causal,sliding_window_64 → comma-separated
    """
    sparse_configs = _resolve_sparse_configs(args)
    shapes = SWEEP_SHAPES[:]
    if args.max_shapes is not None:
        shapes = shapes[:args.max_shapes]

    summary = []
    for config in sparse_configs:
        for shape in shapes:
            B, H, S, D = (int(x) for x in shape.split(","))
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--shape", shape,
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
                "--no-compare" if not args.compare else "",
            ]
            cmd = [c for c in cmd if c]
            label = f"  [{config}] {shape}"

            env = {**os.environ, "SPARSE_CONFIG": config}
            start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.warmup * args.repeat * 30 + 60,  # generous per test
                    env=env,
                )
                elapsed = time.time() - start

                if result.returncode != 0:
                    err = result.stderr.splitlines()[-3:] if result.stderr else ["(no stderr)"]
                    print(f"{label} ... ❌  ERROR ({elapsed:.0f}s)")
                    for line in err:
                        print(f"    {line.strip()}")
                    summary.append((config, shape, "ERROR", None, None, None))
                else:
                    # Parse timing
                    flex_ms = parse_sweep_timing(result.stdout, "Flex Attention")
                    manual_ms = parse_sweep_timing(result.stdout, "Manual Attention")
                    passed = "✅" if flex_ms is not None else "?"
                    print(f"{label} ...  {passed}  flex={flex_ms}  manual={manual_ms}  ({elapsed:.0f}s)")
                    summary.append((config, shape, "OK", flex_ms, manual_ms, None))
            except subprocess.TimeoutExpired:
                print(f"{label} ... ❌  TIMEOUT")
                summary.append((config, shape, "TIMEOUT", None, None, None))
            except Exception as exc:
                print(f"{label} ... ❌  {type(exc).__name__}: {exc}")
                summary.append((config, shape, "ERROR", None, None, type(exc).__name__))

    # Print summary table
    print("\n\n========== SWEEP SUMMARY ==========")
    print(f"{'config':25s} {'shape':16s} {'result':8s} flex_ms manual_ms")
    for config, shape, result, flex_ms, manual_ms, _ in summary:
        flex_s = f"{flex_ms:.3f}" if flex_ms else "-"
        manual_s = f"{manual_ms:.3f}" if manual_ms else "-"
        print(f"  {config:25s} {shape:16s} {result:8s} {flex_s} {manual_s}")


def parse_sweep_timing(stdout, keyword):
    for line in stdout.splitlines():
        if keyword in line and "avg:" in line:
            try:
                parts = line.split("avg:")
                val = parts[1].strip().split("ms")[0].strip()
                val = val.replace("\033[31m", "").replace("\033[0m", "")
                return float(val)
            except (IndexError, ValueError):
                pass
    return None


class TestFlexAttention(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device

    def rundynamictest(self, score_mask_mod, dtype, shape):
        score_mod, mask_mod = score_mask_mod
        args = make_default_args(
            dtype=dtype_name(dtype),
            device=self.device,
            batch=shape[0],
            heads=shape[1],
            seq_len=shape[2],
            head_dim=shape[3],
            warmup=1,
            repeat=1,
        )
        result = run_benchmark(args, score_mod=score_mod, mask_mod=mask_mod)
        self.assertTrue(result["close"])
        return result["outputs"]["flex"], result["outputs"]["manual"]

    # 参数化测试
    def test_builtin_score_mods(self):
        for dtype in test_dtypes:
            for score_mask_mod in test_score_mask_mod_map.items():
                for shape in test_shapes:
                    with self.subTest(dtype=dtype, score_mask_mod=score_mask_mod, shape=shape):
                        self.rundynamictest(score_mask_mod, dtype, shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile Flex Attention versus a manual PyTorch attention baseline.",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "sweep", "profile-target", "msprof", "unittest"],
        default="benchmark",
    )
    parser.add_argument("--target", choices=["both", "flex", "manual"], default="both")
    parser.add_argument("--batch", type=int, default=B)
    parser.add_argument("--heads", type=int, default=H)
    parser.add_argument("--seq-len", type=int, default=S)
    parser.add_argument("--head-dim", type=int, default=D)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Run one shape B,H,S,D or BxHxSxD. Repeat this flag to sweep multiple shapes.",
    )
    parser.add_argument(
        "--shape-suite",
        choices=sorted(SHAPE_SUITES),
        default="single",
        help="Built-in shape sweep. Use smoke for a few representative Flex Attention shapes.",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit how many selected shapes are attempted.",
    )
    parser.add_argument(
        "--stop-on-shape-error",
        dest="continue_on_shape_error",
        action="store_false",
        help="Stop a multi-shape sweep at the first shape failure or OOM.",
    )
    parser.set_defaults(continue_on_shape_error=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=test_device, help="Device to run on, for example auto, npu, npu:0, cpu, or cuda.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument(
        "--manual-mask",
        choices=["precompute", "inside"],
        default="precompute",
        help="precompute excludes dense mask creation from the repeated manual attention loop.",
    )
    parser.add_argument(
        "--sparse-config",
        default=None,
        help="Sparse mask config name from sparse_masks.py (e.g. causal, sliding_window_64).",
    )
    parser.add_argument(
        "--dynamic-compile",
        dest="dynamic_compile",
        action="store_true",
        help="Request dynamic=True for torch.compile. NPU flex attention ignores this unless --allow-npu-dynamic-compile is also set.",
    )
    parser.add_argument(
        "--static-compile",
        dest="dynamic_compile",
        action="store_false",
        help="Pass dynamic=False to torch.compile for fixed-shape experiments. This is the default.",
    )
    parser.set_defaults(dynamic_compile=False)
    parser.add_argument(
        "--allow-npu-dynamic-compile",
        action="store_true",
        help="Force dynamic=True for NPU flex attention. By default it is disabled because it can segfault in torch_npu Inductor.",
    )
    parser.add_argument("--enable-gqa", action="store_true")
    parser.add_argument(
        "--prescale-qk",
        action="store_true",
        help="Set Flex Attention PRESCALE_QK=True. This can be faster but may slightly change numerics.",
    )
    parser.add_argument("--num-warps", type=int, default=None, help="Override Triton num_warps for Flex Attention.")
    parser.add_argument("--num-stages", type=int, default=None, help="Override Triton num_stages for Flex Attention.")
    parser.add_argument("--mstx", action="store_true", help="Emit MSTX ranges around timed loops.")
    parser.add_argument(
        "--suppress-compile-errors",
        action="store_true",
        help="Let torch.compile fall back instead of failing when Inductor compilation errors occur.",
    )
    parser.add_argument("--no-compare", dest="compare", action="store_false")
    parser.set_defaults(compare=True)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--enable-block-reorder", action="store_true",
                        help="Enable block-level query reordering via spectral wave-overlap.")
    parser.add_argument("--block-reorder-mode", default="wave_overlap",
                        choices=sorted(set(["wave_overlap"] + list(REORDER_REGISTRY.keys()))),
                        help="Reorder mode (default: wave_overlap).")
    parser.add_argument("--wave-size", type=int, default=132,
                        help="Wave partition size for reorder algorithms (default: 132).")
    parser.add_argument("--msprof-output", default=None)
    parser.add_argument("--msprof-aic-metrics", default="PipeUtilization")
    parser.add_argument(
        "--msprof-option",
        action="append",
        default=[],
        help="Extra raw msprof option, for example --msprof-option=--task-memory=on",
    )
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be > 0")
    if args.max_shapes is not None and args.max_shapes <= 0:
        parser.error("--max-shapes must be > 0")
    try:
        args.selected_shapes = selected_shapes(args)
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.selected_shapes) == 1:
        args.batch, args.heads, args.seq_len, args.head_dim = args.selected_shapes[0]
    if args.mode in ("profile-target", "msprof") and len(args.selected_shapes) != 1:
        parser.error(f"--mode {args.mode} currently supports exactly one shape")
    if args.mode == "profile-target" and args.target == "both":
        parser.error("--mode profile-target requires --target flex or --target manual")
    return args


if __name__ == "__main__":
    import torch._dynamo

    args = parse_args()
    torch._dynamo.config.suppress_errors = args.suppress_compile_errors

    if args.mode == "benchmark":
        # Support both --sparse-config CLI arg and SPARSE_CONFIG env var
        if args.sparse_config is None and "SPARSE_CONFIG" in os.environ:
            args.sparse_config = os.environ["SPARSE_CONFIG"]
        if args.sparse_config:
            sparse_cfg = get_sparse_config(args.sparse_config)
            score_mod = sparse_cfg.get("score_mod", identity)
            mask_mod = sparse_cfg.get("mask_mod", causal_mask)
            optimizations = sparse_cfg.get("optimizations", None)
            print(f"[sparse-config] {args.sparse_config}: {sparse_cfg.get('description', '')}")

            # Handle special patterns that need pre-built kv_indices
            if sparse_cfg.get("build_block_mask"):
                from sparse_masks import build_random_block_sparse_mask
                extra_args = {"sparse_cfg": sparse_cfg, "build_block_mask_fn": build_random_block_sparse_mask}
                if mask_mod is None:
                    mask_mod = causal_mask  # flex_attention API needs a callable mask_mod
            else:
                extra_args = {}
        else:
            score_mod = identity
            mask_mod = causal_mask
            optimizations = None
            extra_args = {}
        result = run_shape_sweep(args, score_mod=score_mod, mask_mod=mask_mod,
                                 optimizations=optimizations, extra_args=extra_args)
        if result and result.get("close") is False:
            sys.exit(1)
    elif args.mode == "sweep":
        run_sweep_mode(args)
    elif args.mode == "profile-target":
        profile_target(args)
    elif args.mode == "msprof":
        run_msprof(args)
    elif args.mode == "unittest":
        sys.argv = [sys.argv[0]]
        with torch.no_grad():
            unittest.main()
