"""
最小化 sliding_window flex_attention 复现脚本。
对比 flex vs manual，逐步排查 bishengir 崩溃点。
"""
import torch
import math
import os
import sys

# 清理缓存
import shutil
for d in ["/tmp/torchinductor_root", "/tmp/torchinductor_probe"]:
    shutil.rmtree(d, ignore_errors=True)

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
torch.set_float32_matmul_precision("high")

# ── 导入 torch_npu + inductor 补丁 ──
import torch_npu
# 必须在 import torch._inductor.config 之前导入 torch_npu._inductor
# 否则 NPU 的 flex_attention 内核注册不会生效
import torch_npu._inductor  # noqa: F401 — NPU 的 flex_attention 内核补丁
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    BlockMask,
)

B, H, S, D = 1, 4, 512, 64
device = "npu:0"
dtype = torch.bfloat16

print(f"Test: B={B} H={H} S={S} D={D}, device={device}, dtype={dtype}")

# ── 输入 ──
torch.manual_seed(42)
q = torch.randn(B, H, S, D, dtype=dtype, device=device)
k = torch.randn(B, H, S, D, dtype=dtype, device=device)
v = torch.randn(B, H, S, D, dtype=dtype, device=device)
print("inputs created")

# ── mask: sliding_window_64 ──
window_size = 64

def sliding_window_64(batch, head, token_q, token_kv):
    return (token_q - token_kv < window_size) & (token_q >= token_kv)

print("building block mask...")

try:
    block_mask = create_block_mask(
        sliding_window_64,
        1, 1,
        S, S,
        device="cpu",
    ).to(device)
    print(f"block_mask created: kv_num_blocks={block_mask.kv_num_blocks.shape}, "
          f"BLOCK_SIZE={block_mask.BLOCK_SIZE}")
except Exception as e:
    print(f"block_mask creation FAILED: {e}")
    sys.exit(1)

# ── 第1步：eager 模式跑 flex ──
print("\n=== Step 1: flex_attention (eager, no compile) ===")
try:
    with torch.no_grad():
        out_flex_eager = flex_attention(q, k, v, block_mask=block_mask)
    print(f"  ✅ eager flex OK. output shape={out_flex_eager.shape}, any_nan={torch.isnan(out_flex_eager).any().item()}")
except Exception as e:
    print(f"  ❌ eager flex FAILED: {type(e).__name__}: {e}")

# ── 第2步：compile 模式跑 flex ──
print("\n=== Step 2: flex_attention (torch.compile, static) ===")
try:
    compiled_fn = torch.compile(
        lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask),
        backend="inductor",
        dynamic=False,
    )
    print("  compile wrapper created, running first call...")
    with torch.no_grad():
        out_flex_compiled = compiled_fn(q, k, v)
    print(f"  ✅ compiled flex OK. output shape={out_flex_compiled.shape}, any_nan={torch.isnan(out_flex_compiled).any().item()}")
except Exception as e:
    err_msg = str(e)[:500]
    print(f"  ❌ compiled flex FAILED: {type(e).__name__}: {err_msg}")

# ── 第3步：compile + score_mod（无 mask_mod 优化） ──
print("\n=== Step 3: flex_attention with identity score_mod (compile) ===")
try:
    def identity(score, batch, head, token_q, token_kv):
        return score

    compiled_fn2 = torch.compile(
        lambda q, k, v: flex_attention(q, k, v, score_mod=identity, block_mask=block_mask),
        backend="inductor",
        dynamic=False,
    )
    print("  compile wrapper created (score_mod=identity), running first call...")
    with torch.no_grad():
        out_flex_scored = compiled_fn2(q, k, v)
    print(f"  ✅ compiled flex+score_mod OK. output shape={out_flex_scored.shape}")
except Exception as e:
    err_msg = str(e)[:500]
    print(f"  ❌ compiled flex+score_mod FAILED: {type(e).__name__}: {err_msg}")

# ── 第4步：manual attention 对比 ──
print("\n=== Step 4: manual attention (baseline) ===")
try:
    scale = 1.0 / math.sqrt(D)
    row_idx = torch.arange(S, device=device).unsqueeze(1)
    col_idx = torch.arange(S, device=device).unsqueeze(0)
    mask_bool = sliding_window_64(0, 0, row_idx, col_idx)
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    dense_mask = torch.where(mask_bool, zero, neg_inf).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + dense_mask
        attn_weights = torch.softmax(scores, dim=-1)
        out_manual = torch.matmul(attn_weights, v)
    print(f"  ✅ manual OK. output shape={out_manual.shape}")
except Exception as e:
    print(f"  ❌ manual FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

# ── 精度对比 ──
print("\n=== Comparison ===")
if 'out_flex_eager' in dir() and 'out_manual' in dir():
    close = torch.allclose(out_flex_eager.float(), out_manual.float(), rtol=2e-2, atol=2e-2)
    max_diff = (out_flex_eager.float() - out_manual.float()).abs().max().item()
    print(f"  flex(eager) vs manual: allclose={close}, max_abs_diff={max_diff:.6g}")

if 'out_flex_compiled' in dir():
    close2 = torch.allclose(out_flex_compiled.float(), out_manual.float(), rtol=2e-2, atol=2e-2)
    max_diff2 = (out_flex_compiled.float() - out_manual.float()).abs().max().item()
    print(f"  flex(compiled) vs manual: allclose={close2}, max_abs_diff={max_diff2:.6g}")

print("\nDone.")