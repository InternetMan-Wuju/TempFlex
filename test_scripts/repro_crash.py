"""
最小复现：bishengir [ConvertLinalgRToBinary] crash on sliding_window flex_attention.
输出目录: /wyh/code/TempFlex/crash_analysis/
"""
import torch
import torch_npu
import torch_npu._inductor  # noqa: F401
import math, os, shutil, sys
from pathlib import Path
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

OUT = Path("/wyh/code/TempFlex/crash_analysis")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)

# 清理旧缓存
for d in ["/tmp/torchinductor_root", "/tmp/torchinductor_probe", "/root/.triton/dump"]:
    shutil.rmtree(d, ignore_errors=True)

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor_cache")
os.environ["TRITON_CACHE_DIR"] = str(OUT / "triton_cache")
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = str(OUT / "mlir_crash")
OUT.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")

B, H, S, D = 1, 4, 512, 64
device = "npu:0"
dtype = torch.bfloat16

print(f"=== sliding_window_64 crash repro ===")
print(f"B={B} H={H} S={S} D={D}")

torch.manual_seed(42)
q = torch.randn(B, H, S, D, dtype=dtype, device=device)
k = torch.randn(B, H, S, D, dtype=dtype, device=device)
v = torch.randn(B, H, S, D, dtype=dtype, device=device)

window_size = 64
def sliding_window_64(batch, head, token_q, token_kv):
    return (token_q - token_kv < window_size) & (token_q >= token_kv)

block_mask = create_block_mask(sliding_window_64, 1, 1, S, S, device="cpu").to(device)
print(f"block_mask: kv_num_blocks={block_mask.kv_num_blocks.shape}")

# Step 1: eager (should pass)
print("\n[1/3] flex eager ...", end=" ")
with torch.no_grad():
    out_eager = flex_attention(q, k, v, block_mask=block_mask)
print(f"OK shape={out_eager.shape}")

# Step 2: compile — 这步会触发 bishengir crash
print("[2/3] flex torch.compile ...")
compiled_fn = torch.compile(
    lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask),
    backend="inductor", dynamic=False,
)
try:
    with torch.no_grad():
        out_compiled = compiled_fn(q, k, v)
    print(f"  UNEXPECTED: compile succeeded. shape={out_compiled.shape}")
except Exception as e:
    msg = str(e)
    print(f"  EXPECTED: compile CRASHED")
    print(f"  Error type: {type(e).__name__}")
    # 保存完整错误信息
    (OUT / "crash_error.txt").write_text(f"{type(e).__name__}: {msg}\n\nFull:\n{msg}")

# Step 3: 拷贝 triton dump 文件
print("\n[3/3] copying triton dumps...")
for dump_dir in Path("/root/.triton/dump").glob("*"):
    dst = OUT / "triton_dumps" / dump_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in dump_dir.iterdir():
        shutil.copy2(f, dst / f.name)
        print(f"  saved: {dst / f.name}")
    # 也打印 ttgir/mlir 内容的前几行
    for f in dump_dir.glob("*.mlir"):
        print(f"\n--- {f.name} (first 80 lines) ---")
        content = f.read_text()
        lines = content.splitlines()[:80]
        for line in lines:
            print(line)
        print(f"... ({len(content.splitlines())} lines total)")

# Step 4: manual baseline
print("\n[4/4] manual baseline ...", end=" ")
scale = 1.0 / math.sqrt(D)
row_idx = torch.arange(S, device=device).unsqueeze(1)
col_idx = torch.arange(S, device=device).unsqueeze(0)
mask_bool = sliding_window_64(0, 0, row_idx, col_idx)
neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
zero = torch.zeros((), dtype=dtype, device=device)
dense_mask = torch.where(mask_bool, zero, neg_inf).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale + dense_mask
    out_manual = torch.matmul(torch.softmax(scores, dim=-1), v)
print(f"OK shape={out_manual.shape}")
print(f"eager vs manual allclose: {torch.allclose(out_eager.float(), out_manual.float(), rtol=2e-2, atol=2e-2)}")

print(f"\nAll outputs saved to: {OUT}")