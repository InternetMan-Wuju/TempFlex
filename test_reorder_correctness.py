"""Test kernel-internal Q-block reorder correctness.

Kernel-internal reorder changes Q-block PROCESSING ORDER without changing
Q data layout, mask_mod positions, or output ordering. Outputs should be
bitwise identical to standard flex_attention.
"""
import functools
import torch
import sys
sys.path.insert(0, '/wyh/code/TempFlex')
# Import the script's NPU setup function
from flex_attention_run_script import ensure_npu_inductor
ensure_npu_inductor()
from sparse_masks import build_random_block_sparse_mask, identity_score
from torch.nn.attention.flex_attention import BlockMask, flex_attention

def _make_attention(score_mod, block_mask, kernel_options_extra=None):
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}
    if kernel_options_extra:
        kernel_options.update(kernel_options_extra)
    return functools.partial(
        flex_attention, score_mod=score_mod, block_mask=block_mask,
        enable_gqa=False, kernel_options=kernel_options,
    )

def test(seq_len=1024, block_size=128, density=0.3):
    device = 'npu:0'
    B, H, D = 1, 4, 128
    torch.manual_seed(42)
    q = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)

    kv_num, kv_idx, simple_mask = build_random_block_sparse_mask(
        seq_len, block_size=block_size, density=density, seed=42)
    kv_num_bh = kv_num.unsqueeze(0).unsqueeze(0)
    kv_idx_bh = kv_idx.unsqueeze(0).unsqueeze(0)

    # ── Baseline: standard flex_attention ──
    bm = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_bh, kv_indices=kv_idx_bh,
        full_kv_num_blocks=torch.zeros_like(kv_num_bh),
        full_kv_indices=torch.zeros_like(kv_idx_bh),
        BLOCK_SIZE=(block_size, block_size), mask_mod=simple_mask,
    ).to(device)

    baseline_fn = _make_attention(identity_score, bm)
    compiled = torch.compile(baseline_fn, backend="inductor", dynamic=False)
    out_baseline = compiled(q, k, v)
    print(f"Baseline: min={out_baseline.min().item():.4f} max={out_baseline.max().item():.4f} NaN={torch.isnan(out_baseline).sum().item()}")

    # ── Reorder: kernel-internal via PERM in full_q_num_blocks ──
    from torch_npu._inductor.kernel.flex_attention_reorder import compute_and_set_pending_perm
    from torch_npu._inductor.kernel.flex_attention import get_and_clear_pending_perm

    # Rebuild block mask (fresh copy)
    bm2 = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_bh.clone(), kv_indices=kv_idx_bh.clone(),
        full_kv_num_blocks=torch.zeros_like(kv_num_bh),
        full_kv_indices=torch.zeros_like(kv_idx_bh),
        BLOCK_SIZE=(block_size, block_size), mask_mod=simple_mask,
    ).to(device)

    # Compute perm and set side channel
    perm = compute_and_set_pending_perm(
        kv_num_bh, kv_idx_bh,
        torch.zeros_like(kv_num_bh), torch.zeros_like(kv_idx_bh),
        mode="wave_overlap", wave_size=132, verbose=True,
    )

    if perm is None:
        print("SKIP: identity permutation")
        return True

    # Embed PERM into full_q_num_blocks for the lowering hook
    bm2.full_q_num_blocks[0, 0, :] = perm.to(torch.int32)

    reorder_fn = _make_attention(identity_score, bm2)
    compiled2 = torch.compile(reorder_fn, backend="inductor", dynamic=False)
    out_reorder = compiled2(q, k, v)
    print(f"Reorder: min={out_reorder.min().item():.4f} max={out_reorder.max().item():.4f} NaN={torch.isnan(out_reorder).sum().item()}")

    # Clear side channel (cleanup)
    get_and_clear_pending_perm()

    # Compare
    rtol, atol = 0.02, 1e-2
    close = torch.allclose(out_baseline.float(), out_reorder.float(), rtol=rtol, atol=atol)
    max_diff = (out_baseline.float() - out_reorder.float()).abs().max().item()
    print(f"close={close} max_diff={max_diff:.6f}")
    return close

if __name__ == "__main__":
    ok = test()
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'}")
    sys.exit(0 if ok else 1)
