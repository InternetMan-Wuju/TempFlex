"""
Host-side block-level mask builders for flex_attention sparse patterns.

Instead of writing complex mask_mod(b, h, q_idx, kv_idx) functions that
trigger bishengir compiler bugs (bool_to_bool_rintmode, //, %), we build
the block mask entirely on the host (CPU/NPU) and convert it to
FULL_KV_NUM_BLKS / FULL_KV_IDX metadata.

This completely bypasses the mask_mod subgraph — the kernel only needs
block-sparse metadata, no per-token masking.

All builders return: mask[b=1, h=1, MQ, NK] where MQ = ceil(Q_LEN / block_q),
NK = ceil(KV_LEN / block_kv).  True = Q block can attend to KV block.
"""

import math
import torch
from typing import Optional, Callable


# ============================================================
# Core utilities
# ============================================================

def block_mask_to_full_kv(mask: torch.Tensor):
    """Convert bool block mask to FULL_KV metadata.

    Args:
        mask: [Bmask, Hmask, MQ, NK], bool — True means Q block can attend to KV block

    Returns:
        full_kv_num_blks: [Bmask, Hmask, MQ], int32 — valid KV blocks per Q row
        full_kv_idx: [Bmask, Hmask, MQ, NK], int32 — sorted valid KV block indices
    """
    assert mask.dtype == torch.bool
    Bm, Hm, MQ, NK = mask.shape

    full_kv_num_blks = mask.sum(dim=-1).to(torch.int32)

    # argsort: False(valid) first, True(padding) last
    full_kv_idx = torch.argsort(~mask, dim=-1, stable=True).to(torch.int32)

    # Zero out padding positions
    col = torch.arange(NK, device=mask.device).view(1, 1, 1, NK)
    valid = col < full_kv_num_blks[..., None]
    full_kv_idx = torch.where(
        valid, full_kv_idx, torch.zeros_like(full_kv_idx)
    )

    return full_kv_num_blks.contiguous(), full_kv_idx.contiguous()


def empty_partial_metadata(mask: torch.Tensor):
    """Create empty (all-zero) sparse/partial block metadata.

    All blocks go through the FULL_KV path — no per-token masking needed.

    Returns kv_idx with shape (B, H, MQ, NK) to match BlockMask.from_kv_blocks
    expectations for the last dimension size.
    """
    Bm, Hm, MQ, NK = mask.shape
    kv_num_blks = torch.zeros(
        (Bm, Hm, MQ), dtype=torch.int32, device=mask.device
    )
    kv_idx = torch.zeros(
        (Bm, Hm, MQ, NK), dtype=torch.int32, device=mask.device
    )
    return kv_num_blks, kv_idx


def identity_score_mod(score, b, h, q_idx, k_idx):
    """Identity score modifier — no change to attention scores."""
    return score


# ============================================================
# Pattern builders — each returns [1, 1, MQ, NK] bool mask
# ============================================================

def _arange_2d(MQ: int, NK: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create 2D q/k block index tensors."""
    q = torch.arange(MQ, device=device)[:, None]   # [MQ, 1]
    k = torch.arange(NK, device=device)[None, :]    # [1, NK]
    return q, k


def build_block_diagonal(MQ: int, NK: int, device) -> torch.Tensor:
    """Block-diagonal: Q block i only attends to KV block i (causal if MQ==NK)."""
    q, k = _arange_2d(MQ, NK, device)
    mask = q == k
    return mask[None, None, :, :]  # [1, 1, MQ, NK]


def build_checkerboard(
    MQ: int, NK: int, period_blocks: int, device
) -> torch.Tensor:
    """Checkerboard: alternating pattern with period_blocks.

    IMPORTANT: SPARSE_Q_BLOCK_SIZE and SPARSE_KV_BLOCK_SIZE MUST equal
    period_blocks * SPARSE_SUB_BLOCK.  For exact block-alignment, set
    period_blocks=1 and make SPARSE block size match the checkerboard period.
    """
    q, k = _arange_2d(MQ, NK, device)
    mask = ((q + k) % period_blocks) == 0
    return mask[None, None, :, :]


def build_strided(
    MQ: int, NK: int, stride_blocks: int, *, causal: bool = True, device
) -> torch.Tensor:
    """Strided: Q block i attends to KV blocks at i, i-stride, i-2*stride, ...

    Args:
        stride_blocks: stride in block units
        causal: if True, only k <= q
    """
    q, k = _arange_2d(MQ, NK, device)
    dist = q - k
    mask = (dist >= 0) if causal else torch.ones_like(dist, dtype=torch.bool)
    mask = mask & (dist % stride_blocks == 0)
    return mask[None, None, :, :]


def build_dilated_window(
    MQ: int, NK: int,
    radius_blocks: int,
    dilation_blocks: int,
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Dilated sliding window: tokens within a dilated window.

    Args:
        radius_blocks: window radius in block units
        dilation_blocks: dilation factor in block units
        causal: if True, only past tokens
    """
    q, k = _arange_2d(MQ, NK, device)
    if causal:
        dist = q - k
    else:
        dist = torch.abs(q - k)

    mask = (dist >= 0) & (dist <= radius_blocks * dilation_blocks) & (dist % dilation_blocks == 0)
    if causal:
        mask = mask & (q >= k)

    return mask[None, None, :, :]


def build_nested(
    MQ: int, NK: int,
    local_blocks: int,
    stride_blocks: int,
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Nested: local window + strided sampling.

    Masks are OR'd together: local window blocks OR strided blocks.
    """
    q, k = _arange_2d(MQ, NK, device)

    local = torch.abs(q - k) <= local_blocks
    if causal:
        local = local & (q >= k)

    dist = q - k if causal else torch.abs(q - k)
    strided = (dist >= 0) & (dist % stride_blocks == 0)
    if causal:
        strided = strided & (q >= k)

    mask = local | strided
    return mask[None, None, :, :]


def build_uniform_doc(
    MQ: int, NK: int,
    doc_len: int,
    block_q: int,
    *,
    device,
) -> torch.Tensor:
    """Doc-boundary mask: tokens within the same document.

    Document boundaries are aligned to doc_len tokens.
    For block alignment, doc_len MUST be divisible by block_q.
    """
    assert doc_len % block_q == 0, (
        f"doc_len ({doc_len}) must be divisible by block_q ({block_q})"
    )

    # Compute which document each block belongs to
    blocks_per_doc = doc_len // block_q
    doc_id_q = torch.arange(MQ, device=device) // blocks_per_doc           # [MQ]
    doc_id_k = torch.arange(NK, device=device) // blocks_per_doc           # [NK]

    mask = doc_id_q[:, None] == doc_id_k[None, :]  # [MQ, NK]
    return mask[None, None, :, :]


def build_multiscale_dilated(
    MQ: int, NK: int,
    scales: list[tuple[int, int]],  # [(radius, dilation), ...]
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Multi-scale dilated: union of multiple dilated window patterns."""
    q, k = _arange_2d(MQ, NK, device)

    if causal:
        dist = q - k
    else:
        dist = torch.abs(q - k)

    mask = torch.zeros((MQ, NK), dtype=torch.bool, device=device)
    for radius, dilation in scales:
        one = (dist >= 0) & (dist <= radius * dilation) & (dist % dilation == 0)
        if causal:
            one = one & (q >= k)
        mask = mask | one

    return mask[None, None, :, :]


def build_hybrid_sparse(
    MQ: int, NK: int,
    local_blocks: int,
    stride_blocks: int,
    global_every_blocks: int,
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Hybrid: local window + strided + periodic global.

    Masks are OR'd: local OR strided OR global_every.
    """
    q, k = _arange_2d(MQ, NK, device)

    # Local window
    local = torch.abs(q - k) <= local_blocks
    if causal:
        local = local & (q >= k)

    # Strided
    dist = q - k if causal else torch.abs(q - k)
    strided = (dist >= 0) & (dist % stride_blocks == 0)
    if causal:
        strided = strided & (q >= k)

    # Periodic global
    global_mask = (k % global_every_blocks) == 0
    if causal:
        global_mask = global_mask & (q >= k)

    mask = local | strided | global_mask
    return mask[None, None, :, :]


def build_global_local(
    MQ: int, NK: int,
    global_blocks: int,
    local_blocks: int,
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Global + Local: first global_blocks are visible to all, plus a local window."""
    q, k = _arange_2d(MQ, NK, device)

    global_mask = k < global_blocks  # first global_blocks KV blocks visible to all

    local = torch.abs(q - k) <= local_blocks
    if causal:
        local = local & (q >= k)

    # Also: first global_blocks Q blocks can see everything (if causal, up to q)
    q_global_mask = q < global_blocks
    if causal:
        q_global_mask = q_global_mask & (q >= k)

    mask = global_mask | local | q_global_mask
    return mask[None, None, :, :]


def build_sliding_window(
    MQ: int, NK: int,
    window_blocks: int,
    *,
    causal: bool = True,
    device,
) -> torch.Tensor:
    """Sliding window: each Q block can see up to window_blocks KV blocks back."""
    q, k = _arange_2d(MQ, NK, device)
    dist = q - k
    mask = (dist >= 0) & (dist < window_blocks)
    return mask[None, None, :, :]


def build_prefix_lm(
    MQ: int, NK: int,
    prefix_blocks: int,
    device,
) -> torch.Tensor:
    """Prefix LM: first prefix_blocks KV blocks visible to all, plus causal after."""
    q, k = _arange_2d(MQ, NK, device)

    prefix = k < prefix_blocks  # prefix visible to all
    causal = q >= k             # standard causal

    mask = prefix | causal
    return mask[None, None, :, :]


def build_band_global(
    MQ: int, NK: int,
    bandwidth_blocks: int,
    global_blocks: int,
    device,
) -> torch.Tensor:
    """Band + Global: diagonal band + first global_blocks tokens."""
    q, k = _arange_2d(MQ, NK, device)

    dist = q - k
    in_band = (dist <= bandwidth_blocks) & (-dist <= bandwidth_blocks)

    global_mask = k < global_blocks

    mask = in_band | global_mask
    return mask[None, None, :, :]


# ============================================================
# Unified dispatcher
# ============================================================

_PATTERN_BUILDERS: dict[str, Callable[..., torch.Tensor]] = {
    "block_diagonal": build_block_diagonal,
    "checkerboard": build_checkerboard,
    "strided": build_strided,
    "dilated_window": build_dilated_window,
    "nested": build_nested,
    "uniform_doc": build_uniform_doc,
    "multiscale_dilated": build_multiscale_dilated,
    "hybrid_sparse": build_hybrid_sparse,
    "global_local": build_global_local,
    "sliding_window": build_sliding_window,
    "prefix_lm": build_prefix_lm,
    "band_global": build_band_global,
}


def build_block_mask(
    mode: str,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_kv: int,
    device,
    **kwargs,
) -> torch.Tensor:
    """Build block mask for a given sparse pattern.

    Args:
        mode: pattern name (see _PATTERN_BUILDERS)
        q_len: Q sequence length
        kv_len: KV sequence length
        block_q: Q sparse block size
        block_kv: KV sparse block size
        device: target device
        **kwargs: pattern-specific parameters

    Returns:
        mask: [1, 1, MQ, NK] bool tensor
    """
    MQ = (q_len + block_q - 1) // block_q
    NK = (kv_len + block_kv - 1) // block_kv

    builder = _PATTERN_BUILDERS.get(mode)
    if builder is None:
        raise ValueError(
            f"Unknown pattern mode: {mode!r}. Available: {sorted(_PATTERN_BUILDERS.keys())}"
        )

    mask = builder(MQ=MQ, NK=NK, device=device, **kwargs)
    assert mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1, (
        f"Expected [1,1,MQ,NK], got {mask.shape}"
    )
    return mask


def list_patterns() -> list[str]:
    """List all available pattern modes."""
    return sorted(_PATTERN_BUILDERS.keys())
