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
import torch_npu
torch.set_float32_matmul_precision("high")
MSTX_DOMAIN = "flex_attention2"
#---------- 原始 flexattention 相关函数 ----------
def create_attention(score_mod, block_mask, enable_gqa=False, block_m=64, block_n=64):
    kernel_options = {"BLOCK_M": block_m, "BLOCK_N": block_n}
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


def manualattention(q, k, v, mask_mod, dense_mask=None, scale=None):
    """
    用基础 PyTorch 算子实现带 mask 的 Attention。
    maskmod 与原 flexattention 中的定义一致：
    maskmod(batch, head, tokenq, tokenkv) -> bool（True 表示保留，False 表示屏蔽）
    """
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
test_device = "npu"
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
    if str(device).startswith("npu"):
        torch.npu.set_device(device)


def sync_device(device):
    if str(device).startswith("npu"):
        torch.npu.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


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
        dynamic_compile=True,
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
    block_mask = create_block_mask(
        mask_mod,
        1,
        1,
        args.seq_len,
        args.seq_len,
        device=args.device,
    )
    sdpa_fn = create_attention(
        score_mod,
        block_mask=block_mask,
        enable_gqa=args.enable_gqa,
        block_m=args.block_m,
        block_n=args.block_n,
    )
    compiled_sdpa = torch.compile(
        sdpa_fn,
        backend="inductor",
        dynamic=args.dynamic_compile,
    )

    def run():
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
    return torch_npu.npu.mstx.range_start(message, domain=MSTX_DOMAIN)


def mstx_end(range_id, enabled):
    if enabled and range_id is not None:
        torch_npu.npu.mstx.range_end(range_id, domain=MSTX_DOMAIN)


def mstx_mark(message, enabled):
    if enabled:
        torch_npu.npu.mstx.mark(message, domain=MSTX_DOMAIN)


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
    if dtype in (torch.bfloat16, torch.float16):
        return 1e-2, 1e-2
    return 1e-3, 1e-5


def detailed_compare(output_flex, output_manual, rtol, atol, topk=10):
    eps = 1e-6
    flex_f32 = output_flex.float()
    manual_f32 = output_manual.float()
    absdiff = (flex_f32 - manual_f32).abs()
    max_abs = absdiff.max().item()
    mean_abs = absdiff.mean().item()
    median_abs = absdiff.median().item()

    denom = torch.maximum(flex_f32.abs(), manual_f32.abs()).clamp_min(eps)
    rel = absdiff / denom
    max_rel = rel.max().item()

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
        f"max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel * 100:.6g}%, "
        f"mean_abs_diff={mean_abs:.6g}, median_abs_diff={median_abs:.6g}"
    )
    print(f"fail_ratio={fail_ratio * 100:.4f}%  (num_fail={num_fail}/{total})")
    print("--------------------------")

    if topk is not None and topk > 0 and (num_fail > 0 or max_rel > 0.05):
        flat_abs = absdiff.reshape(-1)
        k = min(topk, flat_abs.numel())
        vals, idxs = torch.topk(flat_abs, k)
        batch, heads, seq_len, head_dim = output_manual.shape
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
        "max_rel_diff": max_rel,
        "fail_ratio": fail_ratio,
        "num_fail": num_fail,
        "any_nan_flex": any_nan_flex,
        "any_inf_flex": any_inf_flex,
        "any_nan_manual": any_nan_manual,
        "any_inf_manual": any_inf_manual,
    }


def run_benchmark(args, score_mod=identity, mask_mod=causal_mask):
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
    parser.add_argument("--device", default=test_device)
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
        help="Pass dynamic=True to torch.compile. This is the default.",
    )
    parser.add_argument(
        "--static-compile",
        dest="dynamic_compile",
        action="store_false",
        help="Pass dynamic=False to torch.compile for fixed-shape experiments.",
    )
    parser.set_defaults(dynamic_compile=True)
    parser.add_argument("--enable-gqa", action="store_true")
    parser.add_argument("--mstx", action="store_true", help="Emit MSTX ranges around timed loops.")
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

    # 避免 log 干扰
    torch._dynamo.config.suppress_errors = True
    args = parse_args()

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
