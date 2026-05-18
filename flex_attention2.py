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
)


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
test_device = "auto"
test_dtypes = [torch.bfloat16]
test_score_mask_mod_map = {identity: causal_mask}   # 键为 scoremod，值为 maskmod


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


def make_flex_runner(q, k, v, score_mod, mask_mod, args):
    use_npu = device_is_npu(args.device)
    if use_npu:
        ensure_npu_inductor()

    block_mask_device = "cpu" if use_npu else args.device
    block_mask = create_block_mask(
        mask_mod,
        1,
        1,
        args.seq_len,
        args.seq_len,
        device=block_mask_device,
    ).to(args.device)
    kernel_options_extra = {}
    if mask_mod is causal_mask:
        kernel_options_extra["ROWS_GUARANTEED_SAFE"] = True
        kernel_options_extra["BLOCKS_ARE_CONTIGUOUS"] = True

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
    print(
        f"B:{args.batch} H:{args.heads} S:{args.seq_len} D:{args.head_dim} "
        f"| {label} avg: {avg_ms:.3f} ms "
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
        
        if max_rel_pct > 1:
            print(f"The max diff ratio (a-b)/[(a+b)*2] bigger than 1% ")
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


def run_benchmark(args, score_mod=identity, mask_mod=causal_mask):
    resolve_device(args)
    q, k, v = make_inputs(args)
    outputs = {}
    timings = {}

    if args.target in ("both", "flex"):
        flex_runner = make_flex_runner(q, k, v, score_mod, mask_mod, args)
        outputs["flex"], timings["flex"] = time_runner("Flex Attention", flex_runner, args)

    if args.target in ("both", "manual"):
        manual_runner = make_manual_runner(q, k, v, mask_mod, args)
        outputs["manual"], timings["manual"] = time_runner("Manual Attention", manual_runner, args)

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

    return {
        "outputs": outputs,
        "timings": timings,
        "close": close,
        "stats": stats,
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


class TestFlexAttention(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device

    def rundynamictest(self, score_mask_mod, dtype):
        score_mod, mask_mod = score_mask_mod
        args = make_default_args(dtype=dtype_name(dtype), device=self.device)
        result = run_benchmark(args, score_mod=score_mod, mask_mod=mask_mod)
        self.assertTrue(result["close"])
        return result["outputs"]["flex"], result["outputs"]["manual"]

    # 参数化测试
    def test_builtin_score_mods(self):
        for dtype in test_dtypes:
            for score_mask_mod in test_score_mask_mod_map.items():
                with self.subTest(dtype=dtype, score_mask_mod=score_mask_mod):
                    self.rundynamictest(score_mask_mod, dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile Flex Attention versus a manual PyTorch attention baseline.",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "profile-target", "msprof", "unittest"],
        default="benchmark",
    )
    parser.add_argument("--target", choices=["both", "flex", "manual"], default="both")
    parser.add_argument("--batch", type=int, default=B)
    parser.add_argument("--heads", type=int, default=H)
    parser.add_argument("--seq-len", type=int, default=S)
    parser.add_argument("--head-dim", type=int, default=D)
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
    if args.mode == "profile-target" and args.target == "both":
        parser.error("--mode profile-target requires --target flex or --target manual")
    return args


if __name__ == "__main__":
    import torch._dynamo

    args = parse_args()
    torch._dynamo.config.suppress_errors = args.suppress_compile_errors

    if args.mode == "benchmark":
        result = run_benchmark(args)
        if result["close"] is False:
            sys.exit(1)
    elif args.mode == "profile-target":
        profile_target(args)
    elif args.mode == "msprof":
        run_msprof(args)
    elif args.mode == "unittest":
        sys.argv = [sys.argv[0]]
        with torch.no_grad():
            unittest.main()
