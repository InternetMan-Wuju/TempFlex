# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
# TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_probe \
# python3 flex_attention2.py
# #强制重新编译
# 
import os as _os
_npu_lib = "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib"
_ld = _os.environ.get("LD_LIBRARY_PATH", "")
if _npu_lib not in _ld:
    _os.environ["LD_LIBRARY_PATH"] = f"{_npu_lib}:{_ld}" if _ld else _npu_lib

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
from dataclasses import dataclass
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
        rebuild_block_mask,
        unpermute_output,
        make_reordered_score_mod,
        compute_and_set_pending_perm,
        REORDER_REGISTRY,
    )
    _HAS_REORDER = True
except Exception:
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


@dataclass
class ReorderPlan:
    """Backend-neutral description of a logical block traversal reorder.

    The current external FULL_KV backend materializes this plan as reordered Q
    blocks plus reordered metadata. A future direct mask/score backend should
    consume the same logical plan without changing mask_mod/score_mod token
    semantics.
    """
    q_perm: torch.Tensor
    inv_perm: torch.Tensor
    wave_id: torch.Tensor
    kv_orientation: str
    block_size: int = 128



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


def _full_metadata_to_bool_mask(full_idx, full_num):
    """Reconstruct a row/column bool mask from FULL_KV metadata on CPU."""
    src_idx = full_idx.detach().cpu()
    src_num = full_num.detach().cpu()
    *prefix, n_rows, n_cols = src_idx.shape
    mask = torch.zeros((*prefix, n_rows, n_cols), dtype=torch.bool)
    flat_idx = src_idx.reshape(-1, n_rows, n_cols)
    flat_num = src_num.reshape(-1, n_rows)
    flat_mask = mask.reshape(-1, n_rows, n_cols)
    for outer in range(flat_idx.shape[0]):
        for row in range(n_rows):
            count = int(flat_num[outer, row].item())
            if count <= 0:
                continue
            cols = flat_idx[outer, row, :count].to(torch.long)
            flat_mask[outer, row, cols] = True
    return mask


def _compute_boundary_dp_desc_waves(reordered, wave_size):
    """Choose per-wave KV direction by minimizing adjacent boundary jumps."""
    h_count, n_rows, n_cols = reordered.shape
    wave_size = max(1, int(wave_size))
    n_waves = (n_rows + wave_size - 1) // wave_size
    pad = n_waves * wave_size - n_rows
    if pad > 0:
        reordered = torch.cat(
            [reordered, torch.zeros(h_count, pad, n_cols, dtype=torch.bool)],
            dim=1,
        )

    wave_union = reordered.view(h_count, n_waves, wave_size, n_cols).any(dim=2)
    cols = torch.arange(n_cols).view(1, 1, n_cols)
    lo = torch.where(wave_union, cols, torch.full_like(cols, n_cols)).min(dim=2).values.float()
    hi = torch.where(wave_union, cols, torch.full_like(cols, -1)).max(dim=2).values.float()
    empty = hi < 0
    lo = torch.where(empty, torch.zeros_like(lo), lo)
    hi = torch.where(empty, torch.zeros_like(hi), hi)

    desc_waves = torch.zeros(h_count, n_waves, dtype=torch.bool)
    if n_waves <= 1:
        return desc_waves

    for head in range(h_count):
        start = torch.stack([lo[head], hi[head]], dim=1)
        end = torch.stack([hi[head], lo[head]], dim=1)
        dp = torch.zeros(n_waves, 2, dtype=torch.float32)
        parent = torch.zeros(n_waves, 2, dtype=torch.long)
        for wave in range(1, n_waves):
            cost = dp[wave - 1].view(2, 1) + (
                end[wave - 1].view(2, 1) - start[wave].view(1, 2)
            ).abs()
            dp[wave], parent[wave] = cost.min(dim=0)
        cur = int(dp[-1].argmin().item())
        for wave in range(n_waves - 1, -1, -1):
            desc_waves[head, wave] = bool(cur)
            cur = int(parent[wave, cur].item()) if wave > 0 else 0
    return desc_waves


def _compute_edge_dp_desc_waves(full_idx, full_num, wave_size, edge_blocks=4):
    """Choose per-wave KV direction by maximizing adjacent edge-set overlap."""
    src_idx = full_idx.detach().cpu()
    src_num = full_num.detach().cpu()
    *prefix, n_rows, n_cols = src_idx.shape
    flat_idx = src_idx.reshape(-1, n_rows, n_cols)
    flat_num = src_num.reshape(-1, n_rows)
    h_count = flat_idx.shape[0]
    wave_size = max(1, int(wave_size))
    edge_blocks = max(1, int(edge_blocks))
    n_waves = (n_rows + wave_size - 1) // wave_size

    wave_start = torch.zeros(h_count, n_waves, n_cols, dtype=torch.bool)
    wave_end = torch.zeros(h_count, n_waves, n_cols, dtype=torch.bool)
    for h in range(h_count):
        for row in range(n_rows):
            count = int(flat_num[h, row].item())
            if count <= 0:
                continue
            edge = min(edge_blocks, count)
            wave = row // wave_size
            start_cols = flat_idx[h, row, :edge].to(torch.long)
            end_cols = flat_idx[h, row, count - edge : count].to(torch.long)
            wave_start[h, wave, start_cols] = True
            wave_end[h, wave, end_cols] = True

    desc_waves = torch.zeros(h_count, n_waves, dtype=torch.bool)
    if n_waves <= 1:
        return desc_waves

    for h in range(h_count):
        starts = (wave_start[h], wave_end[h])
        ends = (wave_end[h], wave_start[h])
        dp = torch.zeros(n_waves, 2, dtype=torch.float32)
        parent = torch.zeros(n_waves, 2, dtype=torch.long)
        for wave in range(1, n_waves):
            cost = torch.empty(2, 2, dtype=torch.float32)
            for prev_o in range(2):
                for cur_o in range(2):
                    overlap = (ends[prev_o][wave - 1] & starts[cur_o][wave]).sum().float()
                    cost[prev_o, cur_o] = dp[wave - 1, prev_o] - overlap
            dp[wave], parent[wave] = cost.min(dim=0)
        cur = int(dp[-1].argmin().item())
        for wave in range(n_waves - 1, -1, -1):
            desc_waves[h, wave] = bool(cur)
            cur = int(parent[wave, cur].item()) if wave > 0 else 0
    return desc_waves.reshape(*prefix, n_waves)


def _compute_union_boundary_dp_desc_waves(
    full_idx,
    full_num,
    wave_size,
    edge_blocks=4,
    jump_weight=0.05,
    overlap_weight=0.25,
):
    """Choose KV direction by minimizing wave-to-wave cold edge transitions.

    Unlike boundary_dp's lo/hi interval proxy, this uses the actual first/last
    KV block sets for each wave. This is closer to the patent/FA4 boundary_dp
    idea: preserve the hot KV edge from the previous wave and avoid cold-start
    blocks at the beginning of the next wave.
    """
    src_idx = full_idx.detach().cpu()
    src_num = full_num.detach().cpu()
    *prefix, n_rows, n_cols = src_idx.shape
    flat_idx = src_idx.reshape(-1, n_rows, n_cols)
    flat_num = src_num.reshape(-1, n_rows)
    h_count = flat_idx.shape[0]
    wave_size = max(1, int(wave_size))
    edge_blocks = max(1, int(edge_blocks))
    n_waves = (n_rows + wave_size - 1) // wave_size

    wave_start = torch.zeros(h_count, n_waves, n_cols, dtype=torch.bool)
    wave_end = torch.zeros(h_count, n_waves, n_cols, dtype=torch.bool)
    start_sum = torch.zeros(h_count, n_waves, dtype=torch.float32)
    end_sum = torch.zeros(h_count, n_waves, dtype=torch.float32)
    start_count = torch.zeros(h_count, n_waves, dtype=torch.float32)
    end_count = torch.zeros(h_count, n_waves, dtype=torch.float32)

    for h in range(h_count):
        for row in range(n_rows):
            count = int(flat_num[h, row].item())
            if count <= 0:
                continue
            edge = min(edge_blocks, count)
            wave = row // wave_size
            start_cols = flat_idx[h, row, :edge].to(torch.long)
            end_cols = flat_idx[h, row, count - edge : count].to(torch.long)
            wave_start[h, wave, start_cols] = True
            wave_end[h, wave, end_cols] = True
            start_sum[h, wave] += start_cols.float().sum()
            end_sum[h, wave] += end_cols.float().sum()
            start_count[h, wave] += float(start_cols.numel())
            end_count[h, wave] += float(end_cols.numel())

    start_center = start_sum / start_count.clamp_min(1.0)
    end_center = end_sum / end_count.clamp_min(1.0)
    desc_waves = torch.zeros(h_count, n_waves, dtype=torch.bool)
    if n_waves <= 1:
        return desc_waves.reshape(*prefix, n_waves)

    for h in range(h_count):
        starts = (wave_start[h], wave_end[h])
        ends = (wave_end[h], wave_start[h])
        centers_start = (start_center[h], end_center[h])
        centers_end = (end_center[h], start_center[h])
        dp = torch.zeros(n_waves, 2, dtype=torch.float32)
        parent = torch.zeros(n_waves, 2, dtype=torch.long)
        for wave in range(1, n_waves):
            cost = torch.empty(2, 2, dtype=torch.float32)
            for prev_o in range(2):
                for cur_o in range(2):
                    prev_end = ends[prev_o][wave - 1]
                    cur_start = starts[cur_o][wave]
                    cold = (cur_start & (~prev_end)).sum().float()
                    overlap = (cur_start & prev_end).sum().float()
                    jump = (centers_end[prev_o][wave - 1] - centers_start[cur_o][wave]).abs()
                    cost[prev_o, cur_o] = (
                        dp[wave - 1, prev_o]
                        + cold
                        + float(jump_weight) * jump
                        - float(overlap_weight) * overlap
                    )
            dp[wave], parent[wave] = cost.min(dim=0)
        cur = int(dp[-1].argmin().item())
        for wave in range(n_waves - 1, -1, -1):
            desc_waves[h, wave] = bool(cur)
            cur = int(parent[wave, cur].item()) if wave > 0 else 0
    return desc_waves.reshape(*prefix, n_waves)


def apply_kv_order_to_full_metadata(full_idx, full_num, kv_order, wave_size):
    if kv_order == "asc":
        return full_idx
    if kv_order not in ("desc", "snake", "snake_inv", "boundary_dp", "edge_dp", "union_boundary_dp"):
        raise ValueError(f"Unsupported --kv-order: {kv_order}")

    device = full_idx.device
    *prefix, n_rows, n_cols = full_idx.shape
    col = torch.arange(n_cols, device=device).view(*([1] * (full_idx.ndim - 1)), n_cols)
    valid = col < full_num.unsqueeze(-1)
    src = torch.where(valid, (full_num.unsqueeze(-1) - 1 - col).clamp(min=0), col)
    desc_idx = torch.gather(full_idx, -1, src.to(torch.long))
    if kv_order == "desc":
        return desc_idx
    if kv_order == "boundary_dp":
        mask = _full_metadata_to_bool_mask(full_idx, full_num)
        wave_desc = _compute_boundary_dp_desc_waves(
            mask.reshape(-1, n_rows, n_cols),
            wave_size,
        )
        desc_rows = wave_desc.repeat_interleave(max(1, int(wave_size)), dim=1)[:, :n_rows]
        desc_rows = desc_rows.reshape(*prefix, n_rows, 1).to(device=device)
        return torch.where(desc_rows, desc_idx, full_idx)
    if kv_order == "edge_dp":
        wave_desc = _compute_edge_dp_desc_waves(full_idx, full_num, wave_size, edge_blocks=4)
        desc_rows = wave_desc.repeat_interleave(max(1, int(wave_size)), dim=-1)[..., :n_rows]
        desc_rows = desc_rows.reshape(*prefix, n_rows, 1).to(device=device)
        return torch.where(desc_rows, desc_idx, full_idx)
    if kv_order == "union_boundary_dp":
        wave_desc = _compute_union_boundary_dp_desc_waves(
            full_idx,
            full_num,
            wave_size,
            edge_blocks=4,
        )
        desc_rows = wave_desc.repeat_interleave(max(1, int(wave_size)), dim=-1)[..., :n_rows]
        desc_rows = desc_rows.reshape(*prefix, n_rows, 1).to(device=device)
        return torch.where(desc_rows, desc_idx, full_idx)

    row = torch.arange(n_rows, device=device).view(*([1] * (full_idx.ndim - 2)), n_rows, 1)
    desc_wave = ((row // max(1, int(wave_size))) % 2) == 1
    if kv_order == "snake_inv":
        desc_wave = ~desc_wave
    return torch.where(desc_wave, desc_idx, full_idx)


def apply_npu_reorder_selector(args):
    """Conservative selector for known NPU reorder win/loss regions."""
    sparse_config = getattr(args, "sparse_config", None)
    seq_len = int(getattr(args, "seq_len", 0) or 0)
    disabled = {
        "causal": "causal has natural contiguous KV locality; use causal fastpath instead",
        "dilated_window_bs": "80K repeat tests showed reorder regression",
    }
    if sparse_config in disabled:
        return False, disabled[sparse_config]
    if seq_len >= 65536:
        return False, (
            "skip long-sequence reorder: 80K warmup=5 repeat=20 autotune did not "
            "find a stable >=1.01x candidate"
        )
    stable_32k_modes = {
        "hybrid_sparse_bs": "auction_union_fast",
        "nested_bs": "auction_union_fast",
        "strided_bs": "auction_union_fast",
    }
    if sparse_config in stable_32k_modes and 32768 <= seq_len < 65536:
        args.block_reorder_impl = "external"
        args.block_reorder_mode = stable_32k_modes[sparse_config]
        args.kv_order = "snake_inv"
        return True, f"selected {args.block_reorder_mode} + snake_inv for {sparse_config}@32K repeat=20 fair A/B win"
    return False, f"no validated NPU reorder win for sparse_config={sparse_config!r}, seq_len={seq_len}"


def build_reorder_plan(perm, wave_size, kv_orientation="asc", block_size=128):
    perm_cpu = perm.detach().to(torch.int64).cpu()
    inv_perm = torch.empty_like(perm_cpu)
    inv_perm[perm_cpu] = torch.arange(perm_cpu.numel(), dtype=torch.int64)
    wave_size = max(1, int(wave_size))
    wave_id = torch.arange(perm_cpu.numel(), dtype=torch.int64) // wave_size
    return ReorderPlan(
        q_perm=perm_cpu,
        inv_perm=inv_perm,
        wave_id=wave_id,
        kv_orientation=str(kv_orientation),
        block_size=int(block_size),
    )


def banded_union_wave_reorder(mask_float, wave_size=132, waves_per_block=4, candidate_window=768, select_chunk=1):
    """Host-side port of the patent's NNZ-banded wave KV-union packing."""
    *leading, mb, nb = mask_float.shape
    h_count = 1 if not leading else math.prod(leading)
    mask = mask_float.reshape(h_count, mb, nb).bool()
    device = mask.device
    wave_size = max(1, int(wave_size))
    waves_per_block = max(1, int(waves_per_block))
    macro_size = wave_size * waves_per_block
    candidate_window = max(int(candidate_window), wave_size)
    select_chunk = max(1, int(select_chunk))

    nnz = mask.sum(dim=2).float()
    nnz_order = nnz.argsort(dim=1, descending=True)
    n_macros = (mb + macro_size - 1) // macro_size
    total_rows = n_macros * macro_size
    pad = total_rows - mb
    n_waves = total_rows // wave_size
    if pad > 0:
        pad_idx = torch.arange(mb, mb + pad, device=device).unsqueeze(0).expand(h_count, -1)
        order_pad = torch.cat([nnz_order, pad_idx], dim=1)
    else:
        order_pad = nnz_order

    out_heads = []
    for h in range(h_count):
        m_h = mask[h]
        nnz_h = nnz[h]
        head_waves = []
        for macro in range(n_macros):
            rows = order_pad[h, macro * macro_size : (macro + 1) * macro_size]
            unused = rows < mb
            for _ in range(waves_per_block):
                remaining_pos = torch.where(unused)[0]
                if remaining_pos.numel() == 0:
                    head_waves.append(torch.full((wave_size,), mb, device=device, dtype=torch.long))
                    continue

                seed_pos = remaining_pos[0]
                seed = rows[seed_pos]
                unused[seed_pos] = False
                selected_chunks = [seed.view(1)]
                selected_count = 1
                wave_union = m_h[seed].clone()

                while selected_count < wave_size:
                    remaining_pos = torch.where(unused)[0]
                    if remaining_pos.numel() == 0:
                        break
                    pool_pos = remaining_pos[: min(candidate_window, remaining_pos.numel())]
                    pool = rows[pool_pos]
                    score = (m_h[pool] & (~wave_union).unsqueeze(0)).sum(dim=1).float()
                    take = min(select_chunk, wave_size - selected_count, int(pool.numel()))
                    chosen_local = score.topk(k=take, largest=False, sorted=False).indices
                    chosen_pos = pool_pos[chosen_local]
                    chosen = rows[chosen_pos]
                    unused[chosen_pos] = False
                    selected_chunks.append(chosen)
                    selected_count += take
                    wave_union |= m_h[chosen].any(dim=0)

                wave = torch.cat(selected_chunks, dim=0)
                if wave.numel() < wave_size:
                    wave = torch.cat(
                        [wave, torch.full((wave_size - wave.numel(),), mb, device=device, dtype=torch.long)],
                        dim=0,
                    )
                wave_nnz = torch.where(
                    wave < mb,
                    nnz_h[wave.clamp(max=mb - 1)],
                    torch.zeros_like(wave, dtype=nnz_h.dtype),
                )
                head_waves.append(wave[wave_nnz.argsort(descending=True)])
        out_heads.append(torch.stack(head_waves[:n_waves], dim=0))

    wave_rows = torch.stack(out_heads, dim=0)
    valid = wave_rows < mb
    return wave_rows[valid].view(h_count, mb).reshape(*leading, mb).contiguous()


def banded_union_w8_reorder(mask_float, wave_size=132):
    return banded_union_wave_reorder(mask_float, wave_size=wave_size, waves_per_block=8, candidate_window=1056)


def banded_union_fast_reorder(mask_float, wave_size=132):
    return banded_union_wave_reorder(
        mask_float,
        wave_size=wave_size,
        waves_per_block=4,
        candidate_window=max(2 * int(wave_size), int(wave_size)),
        select_chunk=8,
    )


def wave_union_fast_reorder(mask_float, wave_size=132):
    """Low-overhead row order that clusters rows by KV support interval.

    This is a NPU-friendly approximation of the patent's wave-level KV-union
    objective. It avoids the slow greedy Python loop in banded_union_wave and
    instead sorts rows by the center of their valid KV support, then by NNZ.
    For banded/window-like sparse masks this packs similar KV unions into the
    same wave with O(MB log MB) host work.
    """
    *leading, mb, nb = mask_float.shape
    h_count = 1 if not leading else math.prod(leading)
    mask = mask_float.reshape(h_count, mb, nb).bool()
    device = mask.device
    cols = torch.arange(nb, device=device).view(1, 1, nb)
    nnz = mask.sum(dim=2)
    empty = nnz == 0
    lo = torch.where(mask, cols, torch.full_like(cols, nb)).min(dim=2).values
    hi = torch.where(mask, cols, torch.full_like(cols, -1)).max(dim=2).values
    lo = torch.where(empty, torch.zeros_like(lo), lo)
    hi = torch.where(empty, torch.zeros_like(hi), hi)
    center2 = lo + hi

    # Stable two-pass sort: rows with similar KV span center stay close, and
    # dense rows are placed earlier inside equal-center bands for better wave fill.
    nnz_order = nnz.argsort(dim=1, descending=True, stable=True)
    center_sorted = torch.gather(center2, 1, nnz_order)
    center_rank = center_sorted.argsort(dim=1, stable=True)
    order = torch.gather(nnz_order, 1, center_rank)

    # Keep each wave internally sorted by NNZ descending to preserve the existing
    # wave scheduling property from wave_overlap/fiedler variants.
    wave_size = max(1, int(wave_size))
    n_waves = (mb + wave_size - 1) // wave_size
    pad = n_waves * wave_size - mb
    if pad > 0:
        pad_idx = torch.arange(mb, mb + pad, device=device).view(1, -1).expand(h_count, -1)
        order_padded = torch.cat([order, pad_idx], dim=1)
        nnz_padded = torch.cat([nnz, torch.zeros(h_count, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        order_padded = order
        nnz_padded = nnz
    wave_rows = order_padded.view(h_count, n_waves, wave_size)
    wave_nnz = torch.where(
        wave_rows < mb,
        torch.gather(nnz_padded, 1, wave_rows.clamp(max=mb + pad - 1).reshape(h_count, -1)).view(h_count, n_waves, wave_size),
        torch.zeros_like(wave_rows, dtype=nnz.dtype),
    )
    wave_order = wave_nnz.argsort(dim=2, descending=True, stable=True)
    wave_rows = torch.gather(wave_rows, 2, wave_order)
    valid = wave_rows < mb
    return wave_rows[valid].view(h_count, mb).reshape(*leading, mb).contiguous()


def _mask_rows_to_bitsets(mask_h):
    """Pack one head's block mask rows into Python int bitsets for cheap union costs."""
    mask_cpu = mask_h.detach().cpu().bool()
    mb, nb = mask_cpu.shape
    row_bits = []
    for row in range(mb):
        bits = 0
        cols = torch.where(mask_cpu[row])[0].tolist()
        for col in cols:
            bits |= 1 << int(col)
        row_bits.append(bits)
    return row_bits, nb


def _exact_path_order_for_wave_unions(wave_unions, wave_mean_nnz):
    """Order waves by minimizing cold KV transitions between consecutive unions."""
    n_waves = len(wave_unions)
    if n_waves <= 2:
        return list(range(n_waves))

    full = 1 << n_waves
    # Start from the densest wave; this preserves the existing "heavy rows early"
    # scheduling bias while letting DP optimize the remaining transitions.
    start = max(range(n_waves), key=lambda idx: (wave_mean_nnz[idx], -idx))
    inf = 10**18
    dp = [[inf] * n_waves for _ in range(full)]
    parent = [[-1] * n_waves for _ in range(full)]
    start_mask = 1 << start
    dp[start_mask][start] = int(wave_unions[start].bit_count())

    for state in range(full):
        if not (state & start_mask):
            continue
        for last in range(n_waves):
            cur_cost = dp[state][last]
            if cur_cost >= inf:
                continue
            prev_union = wave_unions[last]
            for nxt in range(n_waves):
                if state & (1 << nxt):
                    continue
                cold_cost = int((wave_unions[nxt] & ~prev_union).bit_count())
                new_state = state | (1 << nxt)
                new_cost = cur_cost + cold_cost
                if new_cost < dp[new_state][nxt]:
                    dp[new_state][nxt] = new_cost
                    parent[new_state][nxt] = last

    state = full - 1
    last = min(range(n_waves), key=lambda idx: dp[state][idx])
    order = []
    while last >= 0:
        order.append(last)
        prev = parent[state][last]
        state &= ~(1 << last)
        last = prev
    order.reverse()
    return order


def auction_union_fast_reorder(
    mask_float,
    wave_size=132,
    group_waves=8,
    candidate_window=None,
    select_chunk=4,
    exact_path=False,
):
    """Patent-style host reorder using KV-union auction inside macro waves.

    Compared with wave_union_fast's interval sort, this directly scores candidate
    rows by how many new KV blocks they add to the current wave union. The masks
    are tiny at block granularity (e.g. 80K/128 -> 640 rows), so Python int
    bitsets keep this path suitable for offline/external metadata generation.
    """
    *leading, mb, nb = mask_float.shape
    h_count = 1 if not leading else math.prod(leading)
    mask = mask_float.reshape(h_count, mb, nb).bool()
    device = mask_float.device
    wave_size = max(1, int(wave_size))
    group_waves = max(1, int(group_waves))
    macro_size = wave_size * group_waves
    candidate_window = int(candidate_window) if candidate_window else macro_size
    candidate_window = max(wave_size, candidate_window)
    select_chunk = max(1, int(select_chunk))

    nnz = mask.sum(dim=2).detach().cpu()
    out_heads = []
    for head in range(h_count):
        row_bits, _ = _mask_rows_to_bitsets(mask[head])
        nnz_h = nnz[head].tolist()
        sorted_rows = sorted(range(mb), key=lambda row: (-nnz_h[row], row))
        used = [False] * mb
        selected_rows = []

        for macro_start in range(0, mb, macro_size):
            macro_waves = []
            macro_limit = min(mb, macro_start + macro_size)
            active_waves = (macro_limit - macro_start + wave_size - 1) // wave_size
            candidate_limit = min(mb, macro_start + candidate_window)

            for wave_idx in range(active_waves):
                rows_left = mb - len(selected_rows) - sum(len(wave) for wave in macro_waves)
                target = min(wave_size, rows_left)
                if target <= 0:
                    break

                wave = []
                wave_union = 0

                seed = None
                for pos in range(macro_start, candidate_limit):
                    row = sorted_rows[pos]
                    if not used[row]:
                        seed = row
                        break
                if seed is None:
                    for row in sorted_rows:
                        if not used[row]:
                            seed = row
                            break
                if seed is None:
                    break

                used[seed] = True
                wave.append(seed)
                wave_union |= row_bits[seed]

                while len(wave) < target:
                    pool = []
                    for pos in range(macro_start, candidate_limit):
                        row = sorted_rows[pos]
                        if not used[row]:
                            # Lower added-union cost is better; NNZ desc and
                            # original row id are stable tie breakers.
                            added = int((row_bits[row] & ~wave_union).bit_count())
                            pool.append((added, -nnz_h[row], row, pos))
                    if not pool:
                        for pos, row in enumerate(sorted_rows):
                            if not used[row]:
                                added = int((row_bits[row] & ~wave_union).bit_count())
                                pool.append((added, -nnz_h[row], row, pos))
                    if not pool:
                        break

                    pool.sort()
                    take = min(select_chunk, target - len(wave), len(pool))
                    for _, _, row, _ in pool[:take]:
                        if used[row]:
                            continue
                        used[row] = True
                        wave.append(row)
                        wave_union |= row_bits[row]

                wave.sort(key=lambda row: (-nnz_h[row], row))
                macro_waves.append(wave)

            if exact_path and len(macro_waves) > 1:
                wave_unions = []
                wave_mean_nnz = []
                for wave in macro_waves:
                    union = 0
                    total = 0.0
                    for row in wave:
                        union |= row_bits[row]
                        total += float(nnz_h[row])
                    wave_unions.append(union)
                    wave_mean_nnz.append(total / max(1, len(wave)))
                wave_order = _exact_path_order_for_wave_unions(wave_unions, wave_mean_nnz)
                macro_waves = [macro_waves[idx] for idx in wave_order]

            for wave in macro_waves:
                selected_rows.extend(wave)

        if len(selected_rows) < mb:
            selected_rows.extend(row for row in sorted_rows if not used[row])
        selected_rows = selected_rows[:mb]
        out_heads.append(torch.tensor(selected_rows, dtype=torch.long, device=device))

    return torch.stack(out_heads, dim=0).reshape(*leading, mb).contiguous()


def auction_union_exact_path_reorder(mask_float, wave_size=132):
    return auction_union_fast_reorder(mask_float, wave_size=wave_size, exact_path=True)


LOCAL_REORDER_REGISTRY = {
    "banded_union_wave": banded_union_wave_reorder,
    "banded_union_wave_w8": banded_union_w8_reorder,
    "banded_union_fast": banded_union_fast_reorder,
    "wave_union_fast": wave_union_fast_reorder,
    "auction_union_fast": auction_union_fast_reorder,
    "auction_union_exact_path": auction_union_exact_path_reorder,
}
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
B, H, S, D = 4, 8, 8192, 128
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
    "large": [
        (1, 4, 4096, 128),
        (1, 4, 8192, 128),
        (2, 4, 8192, 128),
        (2, 8, 16384, 128),
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

    # Support comma-separated multi-device: --device npu:0,npu:1
    device_candidates = [d.strip() for d in requested.split(",") if d.strip()]

    resolved = []
    for device_str in device_candidates:
        if device_str == "auto":
            resolved_device = "npu" if npu_is_available() else "cpu"
        else:
            resolved_device = device_str

        if device_is_npu(resolved_device):
            import_torch_npu(required=True)
            if not npu_is_available():
                raise RuntimeError(
                    f"Requested --device {resolved_device}, but torch.npu.is_available() is False. "
                    "If npu-smi works on the host, check that this Python process/container "
                    "has the Ascend device and runtime mounted."
                )
        resolved.append(resolved_device)

    # Store the full list for multi-device sweeps
    args.devices_list = resolved
    # Keep args.device as the primary device for backward compat
    args.device = resolved[0]

    if len(resolved) > 1:
        print(f"Multi-device mode: {', '.join(resolved)}")

    if requested == "auto":
        print(f"resolved --device auto -> {args.device}")

    return args.device


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
        trim_outliers=True,
        causal_fastpath=True,
        block_reorder_impl="external",
        npu_reorder_selector=False,
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


def make_flex_reorder_runner(q, k, v, args, block_mask, perm):
    """Create a compiled runner that uses torch_npu._flex_attention_reorder custom op.

    The custom op passes PERM as an explicit argument, which allows the Inductor
    lowering to inject it as a kernel argument for kernel-internal Q-block reorder.
    """
    use_npu = device_is_npu(args.device)
    if use_npu:
        ensure_npu_inductor()

    # Unpack block_mask for custom op (8 individual args, not a tuple)
    bm = block_mask
    kv_num_blks = bm.kv_num_blocks
    kv_idxs = bm.kv_indices
    full_kv_num = bm.full_kv_num_blocks
    full_kv_idx = bm.full_kv_indices
    sq_bs = bm.BLOCK_SIZE[0]
    sk_bs = bm.BLOCK_SIZE[1]

    scale = 1.0 / math.sqrt(args.head_dim)

    def reorder_fn(q, k, v, kv_nb, kv_ix, fk_nb, fk_ix, sq, sk, s, p):
        return torch.ops.torch_npu._flex_attention_reorder(
            q, k, v, kv_nb, kv_ix, fk_nb, fk_ix, sq, sk, s, p)

    compiled_fn = torch.compile(reorder_fn, backend="inductor", dynamic=False)

    def run():
        return compiled_fn(
            q, k, v,
            kv_num_blks, kv_idxs, full_kv_num, full_kv_idx,
            sq_bs, sk_bs, scale, perm)

    return run


def make_flex_runner(q, k, v, score_mod, mask_mod, args, block_mask=None, optimizations=None, _graph_salt=None):
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
        if getattr(args, "causal_fastpath", True):
            kernel_options_extra["ROWS_GUARANTEED_SAFE"] = True
            kernel_options_extra["BLOCKS_ARE_CONTIGUOUS"] = True
            print("[fastpath] causal dense fastpath enabled (ROWS_GUARANTEED_SAFE + BLOCKS_ARE_CONTIGUOUS)")
        else:
            print("[fastpath] causal fastpath DISABLED, using generic block-sparse template")
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

    if _graph_salt is not None:
        # Inject _graph_salt into the traced graph to ensure a unique
        # cache key, preventing Inductor from reusing a cached kernel
        # from a different compilation (e.g., baseline's causal fastpath).
        salt = _graph_salt
        orig_sdpa = sdpa_fn
        def salted_fn(q_arg, k_arg, v_arg):
            result = orig_sdpa(q_arg, k_arg, v_arg)
            # Add salt as a no-op: salt.sum()*0 = 0, doesn't change result
            # but forces salt into the FX graph as an additional input
            return result + (salt.sum() * 0.0)
        compiled_sdpa = torch.compile(salted_fn, backend="inductor", dynamic=dynamic_compile)
    else:
        compiled_sdpa = torch.compile(
            sdpa_fn,
            backend="inductor",
            dynamic=dynamic_compile,
        )

    def run():
        return compiled_sdpa(q, k, v)

    return run


def make_manual_runner(q, k, v, mask_mod, args, dense_mask_override=None):
    dense_mask = dense_mask_override
    if dense_mask is None and args.manual_mask == "precompute":
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

        # Per-iteration timing for outlier rejection
        iter_times_ms = []
        for _ in range(args.repeat):
            sync_device(args.device)
            t0 = time.perf_counter()
            last_output = runner()
            sync_device(args.device)
            iter_times_ms.append((time.perf_counter() - t0) * 1000.0)

        mstx_end(range_id, args.mstx)
        mstx_mark(f"profile_repeat_end target={args.target} label={safe_label}", args.mstx)

    # ── Outlier rejection via MAD (Median Absolute Deviation) ──
    n = len(iter_times_ms)
    use_trim = getattr(args, "trim_outliers", True) and n >= 5

    if use_trim:
        sorted_times = sorted(iter_times_ms)
        mid = n // 2
        if n % 2 == 1:
            median = sorted_times[mid]
        else:
            median = (sorted_times[mid - 1] + sorted_times[mid]) / 2.0

        abs_devs = sorted(abs(t - median) for t in iter_times_ms)
        if n % 2 == 1:
            mad = abs_devs[mid]
        else:
            mad = (abs_devs[mid - 1] + abs_devs[mid]) / 2.0

        threshold = 3.0 * mad
        if threshold == 0.0:
            # All iterations identical — no outliers possible
            kept = iter_times_ms
        else:
            kept = [t for t in iter_times_ms if abs(t - median) <= threshold]

        dropped = n - len(kept)
        if len(kept) == 0:
            avg_ms = median
        else:
            avg_ms = sum(kept) / len(kept)
    else:
        avg_ms = sum(iter_times_ms) / n
        dropped = 0

    RED = "\033[31m"
    RESET = "\033[0m"

    trim_note = f", trimmed: {dropped}/{n} outliers dropped" if (use_trim and dropped > 0) else ""
    print(
        f"B:{args.batch} H:{args.heads} S:{args.seq_len} D:{args.head_dim} "
        f"| {label} avg: {RED}{avg_ms:.3f} ms{RESET} "
        f"(warmup={args.warmup}, repeat={args.repeat}{trim_note})"
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
        use_pure_block_sparse = False
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
        elif extra_args.get("use_full_kv_metadata"):
            # Build block-level mask on host, convert to FULL_KV metadata.
            # Also build token-level dense mask for manual correctness check.
            from pattern_to_block_mask import (
                build_block_mask as _build_bm,
                block_mask_to_full_kv,
                empty_partial_metadata,
                block_mask_to_token_mask,
            )
            params = extra_args["block_mask_params"].copy()
            mode = params.pop("mode")
            block_q = 128  # SPARSE_Q_BLOCK_SIZE
            block_kv = 128  # SPARSE_KV_BLOCK_SIZE
            block_mask_device = "cpu" if device_is_npu(args.device) else args.device
            mask = _build_bm(
                mode=mode, q_len=args.seq_len, kv_len=args.seq_len,
                block_q=block_q, block_kv=block_kv,
                device=block_mask_device, **params,
            )
            density = mask.float().mean().item()
            print(f"[block mask] {mode}: density={density:.2%} shape={list(mask.shape)}")
            full_num, full_idx = block_mask_to_full_kv(mask)
            kv_num, kv_idx = empty_partial_metadata(mask)
            # Build token-level dense mask for manual attention comparison
            # (same pattern, so flex vs manual is a fair comparison)
            token_dense_mask = block_mask_to_token_mask(
                mask, block_q, block_kv, args.seq_len, args.seq_len,
            ).to(args.device)
            extra_args["_token_dense_mask"] = token_dense_mask
            # Use a simple causal mask_mod for the API (not actually used
            # since all blocks are FULL and PURE_BLOCK_SPARSE skips it)
            block_mask_override = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num,
                kv_indices=kv_idx,
                full_kv_num_blocks=full_num,
                full_kv_indices=full_idx,
                BLOCK_SIZE=(block_q, block_kv),
                mask_mod=mask_mod,
            ).to(args.device)
            bm_opts = dict(optimizations or {})
            bm_opts["PURE_BLOCK_SPARSE"] = True
            flex_runner = make_flex_runner(q.clone(), k.clone(), v.clone(), score_mod, mask_mod, args,
                                           block_mask=block_mask_override,
                                           optimizations=bm_opts)
        else:
            flex_runner = make_flex_runner(q, k, v, score_mod, mask_mod, args,
                                           optimizations=optimizations)
        outputs["flex"], timings["flex"] = time_runner("Flex Attention", flex_runner, args)

    if (
        args.target in ("both", "manual")
        or (args.target == "reorder" and args.compare)
    ) and mask_mod is not None:
        if extra_args.get("use_full_kv_metadata") and "_token_dense_mask" not in extra_args:
            from pattern_to_block_mask import (
                build_block_mask as _build_bm,
                block_mask_to_token_mask,
            )
            params = extra_args["block_mask_params"].copy()
            mode = params.pop("mode")
            block_q = 128
            block_kv = 128
            block_mask_device = "cpu" if device_is_npu(args.device) else args.device
            mask_for_manual = _build_bm(
                mode=mode,
                q_len=args.seq_len,
                kv_len=args.seq_len,
                block_q=block_q,
                block_kv=block_kv,
                device=block_mask_device,
                **params,
            )
            extra_args["_token_dense_mask"] = block_mask_to_token_mask(
                mask_for_manual,
                block_q,
                block_kv,
                args.seq_len,
                args.seq_len,
            ).to(args.device)
        manual_dense = extra_args.get("_token_dense_mask")
        manual_runner = make_manual_runner(q, k, v, mask_mod, args, dense_mask_override=manual_dense)
        outputs["manual"], timings["manual"] = time_runner("Manual Attention", manual_runner, args)

    # ── Reorder variant: patent-style external Q/mask reorder by default ──
    reorder_hit_rate = None
    reorder_enabled = getattr(args, "enable_block_reorder", False)
    if reorder_enabled and getattr(args, "npu_reorder_selector", False):
        reorder_enabled, selector_reason = apply_npu_reorder_selector(args)
        print(f"[reorder selector] {selector_reason}")
    if reorder_enabled and args.sparse_config == "causal":
        print("[reorder] causal sparse-config skipped: causal has natural contiguous KV locality; reorder target is non-causal block-sparse patterns")
        reorder_enabled = False
    if (
        reorder_enabled
        and args.sparse_config != "causal"
        and _HAS_REORDER
        and device_is_npu(args.device)
    ):
        # Clear any stale side-channel state
        from torch_npu._inductor.kernel.flex_attention import (
            get_and_clear_pending_perm,
            set_pending_perm,
        )
        get_and_clear_pending_perm()
        # Reset dynamo to prevent compilation deadlock when the reorder
        # produces a different template than the baseline.
        torch._dynamo.reset()

        block_mask_device = "cpu" if device_is_npu(args.device) else args.device
        reorder_block_mask = block_mask_override if "block_mask_override" in locals() else None
        mask_float = None
        mask = None

        # Build block mask for reorder perm computation
        if args.sparse_config == "causal":
            from pattern_to_block_mask import (
                build_block_mask as _build_bm,
                block_mask_to_full_kv,
                empty_partial_metadata,
            )
            block_q = 128
            block_kv = 128
            mask = _build_bm(
                mode="causal",
                q_len=args.seq_len,
                kv_len=args.seq_len,
                block_q=block_q,
                block_kv=block_kv,
                device=block_mask_device,
            )
            n_q = mask.shape[2]
            n_kv = mask.shape[3]
            mask_float = mask[0, 0].float()
            full_num, full_idx = block_mask_to_full_kv(mask)
            kv_num, kv_idx = empty_partial_metadata(mask)
            reorder_block_mask = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num.to(args.device),
                kv_indices=kv_idx.to(args.device),
                full_kv_num_blocks=full_num.to(args.device),
                full_kv_indices=full_idx.to(args.device),
                BLOCK_SIZE=(block_q, block_kv),
                mask_mod=mask_mod,
            ).to(args.device)
        elif extra_args.get("use_full_kv_metadata"):
            # FULL_KV metadata pattern: build block mask from pattern builder
            from pattern_to_block_mask import (
                build_block_mask as _build_bm,
                block_mask_to_full_kv,
                empty_partial_metadata,
            )
            params = extra_args["block_mask_params"].copy()
            mode = params.pop("mode")
            block_q = 128
            block_kv = 128
            mask = _build_bm(
                mode=mode, q_len=args.seq_len, kv_len=args.seq_len,
                block_q=block_q, block_kv=block_kv,
                device=block_mask_device, **params,
            )
            # Build dense mask for perm computation
            n_q = mask.shape[2]
            n_kv = mask.shape[3]
            mask_float = mask[0, 0].float()  # [MQ, NK]
            if reorder_block_mask is None:
                full_num, full_idx = block_mask_to_full_kv(mask)
                kv_num, kv_idx = empty_partial_metadata(mask)
                reorder_block_mask = BlockMask.from_kv_blocks(
                    kv_num_blocks=kv_num.to(args.device),
                    kv_indices=kv_idx.to(args.device),
                    full_kv_num_blocks=full_num.to(args.device),
                    full_kv_indices=full_idx.to(args.device),
                    BLOCK_SIZE=(block_q, block_kv),
                    mask_mod=mask_mod,
                ).to(args.device)
        elif extra_args.get("build_block_mask_fn") and args.sparse_config:
            kv_num, kv_idx, _ = extra_args["build_block_mask_fn"](args.seq_len)
            kv_num_bh = kv_num.unsqueeze(0).unsqueeze(0)
            kv_idx_bh = kv_idx.unsqueeze(0).unsqueeze(0)
            full_kv_num = torch.zeros_like(kv_num_bh)
            full_kv_idx = torch.zeros_like(kv_idx_bh)
            bm = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_bh, kv_indices=kv_idx_bh,
                full_kv_num_blocks=full_kv_num, full_kv_indices=full_kv_idx,
                BLOCK_SIZE=(128, 128), mask_mod=mask_mod,
            ).to(args.device)
            reorder_block_mask = bm
            mask_float = rebuild_block_mask(
                bm.kv_num_blocks.cpu(), bm.kv_indices.cpu(),
                bm.kv_indices.shape[2], bm.kv_indices.shape[3],
                full_kv_num_blocks=bm.full_kv_num_blocks.cpu() if bm.full_kv_num_blocks is not None else None,
                full_kv_indices=bm.full_kv_indices.cpu() if bm.full_kv_indices is not None else None,
                device="cpu",
            )
            n_q = mask_float.shape[1]
            n_kv = mask_float.shape[2]
        else:
            if reorder_block_mask is None:
                reorder_block_mask = create_block_mask(
                    mask_mod, 1, 1, args.seq_len, args.seq_len,
                    device=block_mask_device,
                ).to(args.device)
            bm = reorder_block_mask
            mask_float = rebuild_block_mask(
                bm.kv_num_blocks.cpu(), bm.kv_indices.cpu(),
                bm.kv_indices.shape[2], bm.kv_indices.shape[3],
                full_kv_num_blocks=bm.full_kv_num_blocks.cpu() if bm.full_kv_num_blocks is not None else None,
                full_kv_indices=bm.full_kv_indices.cpu() if bm.full_kv_indices is not None else None,
                device="cpu",
            )
            n_q = mask_float.shape[1]
            n_kv = mask_float.shape[2]

        # Compute baseline hit rate (for reference, before reorder)
        baseline_hit = 0.0
        if extra_args.get("use_full_kv_metadata"):
            # Hit rate from block mask directly
            if mask.numel() > 1:
                # Simple heuristic: fraction of adjacent blocks that are both valid
                hits = 0
                total = 0
                mask_np = mask[0, 0].cpu().numpy()
                for i in range(min(n_q, mask_np.shape[0])):
                    for j in range(min(n_kv - 1, mask_np.shape[1] - 1)):
                        if mask_np[i, j] and mask_np[i, j + 1]:
                            hits += 1
                        if mask_np[i, j]:
                            total += 1
                baseline_hit = hits / max(total, 1)
        else:
            baseline_hit = compute_block_hit_rate(
                reorder_block_mask.kv_indices, reorder_block_mask.kv_num_blocks,
                reorder_block_mask.full_kv_indices, reorder_block_mask.full_kv_num_blocks,
            )

        t0 = time.perf_counter()
        try:
            if args.block_reorder_mode == "identity":
                perm = torch.arange(mask_float.shape[-2], dtype=torch.int32)
                perm_2d = perm.unsqueeze(0)
            else:
                reorder_fn = LOCAL_REORDER_REGISTRY.get(args.block_reorder_mode)
                if reorder_fn is None:
                    reorder_fn = REORDER_REGISTRY.get(args.block_reorder_mode)
                if reorder_fn is None and args.block_reorder_mode == "wave_overlap":
                    from torch_npu._inductor.kernel.flex_attention_reorder import (
                        wave_overlap_reorder as reorder_fn,
                    )
                if reorder_fn is None:
                    raise ValueError(f"Unknown reorder mode: {args.block_reorder_mode}")
                perm_2d = reorder_fn(mask_float.unsqueeze(0), wave_size=args.wave_size)
                if isinstance(perm_2d, tuple):
                    perm_2d = perm_2d[0]
                perm = perm_2d[0].to(torch.int32)
            is_identity = torch.equal(perm, torch.arange(len(perm), dtype=torch.int32))
            if is_identity and not getattr(args, "allow_identity_reorder", False):
                print("[reorder] Identity permutation — reorder skipped")
                perm = None
            elif is_identity:
                print(f"[reorder] Identity permutation forced ({len(perm)} blocks)")
            else:
                print(f"[reorder] Non-identity permutation computed ({len(perm)} blocks)")
        except Exception as e:
            print(f"[reorder] Error computing permutation: {e}")
            import traceback; traceback.print_exc()
            perm = None
        reorder_comp_ms = (time.perf_counter() - t0) * 1000.0

        if perm is not None:
            reorder_args = SimpleNamespace(**vars(args))
            if reorder_args.block_reorder_impl == "internal":
                if reorder_args.causal_fastpath:
                    print("[reorder] disabling causal fastpath for reorder run so PERM template can be selected")
                    reorder_args.causal_fastpath = False
                reorder_opts = dict(optimizations or {})
                if extra_args.get("use_full_kv_metadata") or args.sparse_config == "causal":
                    reorder_opts["PURE_BLOCK_SPARSE"] = True
                if args.sparse_config == "causal":
                    reorder_opts["PURE_BLOCK_SPARSE_CAUSAL"] = True
                if args.wave_size > 1:
                    reorder_opts["WAVE_SIZE"] = args.wave_size
                    reorder_opts["ENABLE_REORDER"] = True
                    if getattr(args, "wave_single_tile_debug", False):
                        reorder_opts["WAVE_SINGLE_TILE_DEBUG"] = True
                else:
                    set_pending_perm(perm.unsqueeze(0))
                reorder_block_mask_for_run = reorder_block_mask
                if reorder_opts.get("PURE_BLOCK_SPARSE", False) and args.wave_size > 1:
                    actual_kv_num = reorder_block_mask.kv_num_blocks
                    actual_kv_idx = reorder_block_mask.kv_indices
                    if (
                        reorder_block_mask.full_kv_num_blocks is not None
                        and reorder_block_mask.full_kv_indices is not None
                        and int(reorder_block_mask.full_kv_num_blocks.max().item()) > 0
                    ):
                        actual_kv_num = reorder_block_mask.full_kv_num_blocks
                        actual_kv_idx = reorder_block_mask.full_kv_indices
                    reorder_block_mask_for_run = BlockMask.from_kv_blocks(
                        kv_num_blocks=actual_kv_num,
                        kv_indices=actual_kv_idx,
                        full_kv_num_blocks=torch.zeros_like(actual_kv_num),
                        full_kv_indices=torch.zeros_like(actual_kv_idx),
                        BLOCK_SIZE=reorder_block_mask.BLOCK_SIZE,
                        mask_mod=mask_mod,
                    ).to(args.device)
                if reorder_opts.get("PURE_BLOCK_SPARSE", False) and args.wave_size <= 1:
                    actual_kv_num = reorder_block_mask.kv_num_blocks
                    actual_kv_idx = reorder_block_mask.kv_indices
                    if (
                        reorder_block_mask.full_kv_num_blocks is not None
                        and reorder_block_mask.full_kv_indices is not None
                        and int(reorder_block_mask.full_kv_num_blocks.max().item()) > 0
                    ):
                        actual_kv_num = reorder_block_mask.full_kv_num_blocks
                        actual_kv_idx = reorder_block_mask.full_kv_indices

                    perm_prefix = torch.zeros_like(actual_kv_idx)
                    if args.wave_size > 1:
                        perm_prefix.view(-1)[: perm.numel()] = perm.to(
                            dtype=torch.int32,
                            device=actual_kv_idx.device,
                        )
                    else:
                        sparse_q_multiple = reorder_block_mask.BLOCK_SIZE[0] // args.block_m
                        tile_perm_vals = []
                        for src_sparse in perm.tolist():
                            for intra in range(sparse_q_multiple):
                                tile_perm_vals.append(src_sparse * sparse_q_multiple + intra)
                        tile_perm = torch.tensor(
                            tile_perm_vals,
                            dtype=torch.int32,
                            device=actual_kv_idx.device,
                        )
                        perm_prefix.view(-1)[: tile_perm.numel()] = tile_perm
                    zeros_full_num = torch.zeros_like(actual_kv_num)
                    reorder_block_mask_for_run = BlockMask.from_kv_blocks(
                        kv_num_blocks=actual_kv_num,
                        kv_indices=actual_kv_idx,
                        full_kv_num_blocks=zeros_full_num,
                        full_kv_indices=perm_prefix,
                        BLOCK_SIZE=reorder_block_mask.BLOCK_SIZE,
                        mask_mod=mask_mod,
                    ).to(args.device)
                reorder_runner = make_flex_runner(
                    q.clone(), k.clone(), v.clone(),
                    score_mod, mask_mod, reorder_args,
                    block_mask=reorder_block_mask_for_run,
                    optimizations=reorder_opts if reorder_opts else None,
                    _graph_salt=torch.tensor([1], device=q.device),
                )
                outputs["flex_reorder"], timings["flex_reorder"] = time_runner(
                    f"Flex+{args.block_reorder_mode}+internal", reorder_runner, reorder_args)
                reorder_hit_rate = (baseline_hit, baseline_hit)
                print(
                    f"  ── reorder computation time (CPU): {reorder_comp_ms:.3f} ms "
                    f"| kernel time: {timings.get('flex_reorder', 0):.3f} ms "
                    f"| comp/kernel ratio: {reorder_comp_ms / max(timings.get('flex_reorder', 1e-6), 1e-6):.2%}"
                )
                print("  ── using kernel-internal PERM; metadata hit-rate metric is unchanged by design")
                timings["reorder_comp"] = reorder_comp_ms
            else:
                # Patent-style external path, matching the sparse-attn reference:
                # gather Q blocks by perm, gather sparse metadata rows by the same
                # perm, run a stable single-tile block-sparse kernel, then invert.
                B, H, S, D = q.shape
                n_blocks = len(perm)
                block_size = 128
                dev = q.device
                reorder_plan = build_reorder_plan(
                    perm,
                    args.wave_size,
                    getattr(reorder_args, "kv_order", "asc"),
                    block_size=block_size,
                )

                perm_dev = reorder_plan.q_perm.to(dev)
                inv_perm_dev = reorder_plan.inv_perm.to(dev)

                q_blocks = q.view(B, H, n_blocks, block_size, D)
                perm_expanded = perm_dev.view(1, 1, n_blocks, 1, 1).expand(B, H, n_blocks, block_size, D)
                q_reordered = torch.gather(q_blocks, 2, perm_expanded).reshape(B, H, S, D)

                if extra_args.get("use_full_kv_metadata"):
                    mask_reordered = mask[0, 0][perm_dev.cpu()]
                    mask_reordered = mask_reordered.unsqueeze(0).unsqueeze(0)
                    full_num_reordered, full_idx_reordered = block_mask_to_full_kv(mask_reordered)
                    full_idx_reordered = apply_kv_order_to_full_metadata(
                        full_idx_reordered,
                        full_num_reordered,
                        reorder_plan.kv_orientation,
                        args.wave_size,
                    )
                    kv_num_reordered, kv_idx_reordered = empty_partial_metadata(mask_reordered)
                    bm_reordered = BlockMask.from_kv_blocks(
                        kv_num_blocks=kv_num_reordered.to(dev),
                        kv_indices=kv_idx_reordered.to(dev),
                        full_kv_num_blocks=full_num_reordered.to(dev),
                        full_kv_indices=full_idx_reordered.to(dev),
                        BLOCK_SIZE=(block_size, block_size),
                        mask_mod=mask_mod,
                    ).to(dev)
                else:
                    bm = reorder_block_mask
                    kv_idx_perm = perm_dev.view(1, 1, n_blocks, 1).expand(1, 1, n_blocks, bm.kv_indices.shape[-1])
                    kv_indices_reordered = torch.gather(bm.kv_indices, 2, kv_idx_perm)
                    kv_num_perm = perm_dev.view(1, 1, n_blocks).expand(1, 1, n_blocks)
                    kv_num_reordered = torch.gather(bm.kv_num_blocks, 2, kv_num_perm)
                    zeros_kv_num = torch.zeros_like(kv_num_reordered)
                    zeros_kv_idx = torch.zeros_like(kv_indices_reordered)
                    bm_reordered = BlockMask.from_kv_blocks(
                        kv_num_blocks=zeros_kv_num,
                        kv_indices=zeros_kv_idx,
                        full_kv_num_blocks=kv_num_reordered,
                        full_kv_indices=kv_indices_reordered,
                        BLOCK_SIZE=bm.BLOCK_SIZE,
                        mask_mod=mask_mod,
                    ).to(dev)

                reorder_opts = {"PURE_BLOCK_SPARSE": True}
                reorder_runner = make_flex_runner(
                    q_reordered.clone(), k.clone(), v.clone(),
                    score_mod, mask_mod, reorder_args,
                    block_mask=bm_reordered,
                    optimizations=reorder_opts,
                )
                outputs["flex_reorder"], timings["flex_reorder"] = time_runner(
                    f"Flex+{args.block_reorder_mode}+external", reorder_runner, reorder_args)

                out_reordered = outputs["flex_reorder"]
                out_blocks = out_reordered.view(B, H, n_blocks, block_size, out_reordered.shape[-1])
                inv_expanded = inv_perm_dev.view(1, 1, n_blocks, 1, 1).expand(B, H, n_blocks, block_size, out_blocks.shape[-1])
                out_unperm = torch.gather(out_blocks, 2, inv_expanded)
                outputs["flex_reorder"] = out_unperm.reshape(B, H, S, out_reordered.shape[-1])
                reorder_hit_rate = (baseline_hit, baseline_hit)
                print(
                    f"  ── reorder computation time (CPU): {reorder_comp_ms:.3f} ms "
                    f"| kernel time: {timings.get('flex_reorder', 0):.3f} ms "
                    f"| comp/kernel ratio: {reorder_comp_ms / max(timings.get('flex_reorder', 1e-6), 1e-6):.2%}"
                )
                timings["reorder_comp"] = reorder_comp_ms
        else:
            reorder_hit_rate = (baseline_hit, baseline_hit)
            print("[reorder] Identity permutation — reorder skipped")

    close = None
    stats = None
    if args.compare and "manual" in outputs:
        dtype = dtype_from_name(args.dtype)
        rtol, atol = tolerance_for(dtype, args.rtol, args.atol)
        print(f"rtol={rtol}, atol={atol}")
    if args.target == "both" and args.compare and "flex" in outputs and "manual" in outputs:
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

    # ── Reorder vs manual comparison ──
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
        stats = reorder_stats
        close = close_reorder

        # Extra NaN analysis for reorder output
        o_reorder = outputs["flex_reorder"]
        if torch.isnan(o_reorder).any():
            nan_mask = torch.isnan(o_reorder)
            nan_count = nan_mask.sum().item()
            total = o_reorder.numel()
            print(f"  NaN count: {nan_count}/{total} ({100.0*nan_count/total:.2f}%)")
            nan_per_pos = nan_mask.any(dim=-1).any(dim=1).float().mean(dim=0)
            nan_seq_positions = torch.where(nan_per_pos > 0)[0]
            print(f"  Number of sequence positions with any NaN: {len(nan_seq_positions)}/{o_reorder.shape[2]}")
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

    resolve_device(args)  # populates args.devices_list

    shapes = selected_shapes(args)
    if len(shapes) == 1 and len(args.devices_list) == 1:
        return run_benchmark(args_for_shape(args, shapes[0]), score_mod=score_mod, mask_mod=mask_mod,
                             optimizations=optimizations, extra_args=extra_args)

    shape_results = []
    n_devices = len(args.devices_list)
    device_tag = f" across {n_devices} devices" if n_devices > 1 else ""
    print(f"running shape sweep: {len(shapes)} shapes{device_tag}")

    for device in args.devices_list:
        for index, shape in enumerate(shapes, start=1):
            shape_args = args_for_shape(args, shape)
            shape_args.device = device  # pin to specific device
            device_label = f" [{device}]" if n_devices > 1 else ""
            print(
                f"\n=== shape {index}/{len(shapes)}: "
                f"B:{shape[0]} H:{shape[1]} S:{shape[2]} D:{shape[3]}{device_label} ==="
            )
            try:
                result = run_benchmark(shape_args, score_mod=score_mod, mask_mod=mask_mod,
                                        optimizations=optimizations, extra_args=extra_args)
                shape_results.append((shape, device, result, None))
            except RuntimeError as exc:
                message = str(exc).splitlines()[0]
                print(f"shape failed: {type(exc).__name__}: {message}")
                shape_results.append((shape, device, None, exc))
                if not args.continue_on_shape_error:
                    raise
            finally:
                release_device_memory(shape_args.device)

    print("\n-------- shape sweep summary --------")
    failed = False
    for shape, device, result, error in shape_results:
        device_col = f" [{device}]" if n_devices > 1 else ""
        label = f"B:{shape[0]} H:{shape[1]} S:{shape[2]} D:{shape[3]}{device_col}"
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


def prepare_sparse_execution(args):
    # Support both --sparse-config CLI arg and SPARSE_CONFIG env var.
    if args.sparse_config is None and "SPARSE_CONFIG" in os.environ:
        args.sparse_config = os.environ["SPARSE_CONFIG"]
    if not args.sparse_config:
        return identity, causal_mask, None, {}

    sparse_cfg = get_sparse_config(args.sparse_config)
    score_mod = sparse_cfg.get("score_mod", identity)
    mask_mod = sparse_cfg.get("mask_mod", causal_mask)
    optimizations = sparse_cfg.get("optimizations", None)
    print(f"[sparse-config] {args.sparse_config}: {sparse_cfg.get('description', '')}")

    if sparse_cfg.get("build_block_mask"):
        from sparse_masks import build_random_block_sparse_mask
        extra_args = {
            "sparse_cfg": sparse_cfg,
            "build_block_mask_fn": build_random_block_sparse_mask,
        }
        if mask_mod is None:
            mask_mod = causal_mask
    elif sparse_cfg.get("use_full_kv_metadata"):
        extra_args = {
            "sparse_cfg": sparse_cfg,
            "use_full_kv_metadata": True,
            "block_mask_params": sparse_cfg.get("block_mask_params", {}).copy(),
        }
        if mask_mod is None:
            mask_mod = causal_mask
    else:
        extra_args = {}
    return score_mod, mask_mod, optimizations, extra_args


def profile_target(args):
    resolve_device(args)
    print(
        f"profile target={args.target}, pid={os.getpid()}, "
        f"manual_mask={args.manual_mask}, dynamic_compile={args.dynamic_compile}, "
        f"mstx={args.mstx}"
    )
    args.compare = False
    score_mod, mask_mod, optimizations, extra_args = prepare_sparse_execution(args)
    run_benchmark(
        args,
        score_mod=score_mod,
        mask_mod=mask_mod,
        optimizations=optimizations,
        extra_args=extra_args,
    )


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
    if args.sparse_config:
        argv.extend(["--sparse-config", args.sparse_config])
    if args.enable_block_reorder:
        argv.append("--enable-block-reorder")
        argv.extend(["--block-reorder-impl", args.block_reorder_impl])
        argv.extend(["--block-reorder-mode", args.block_reorder_mode])
        argv.extend(["--wave-size", str(args.wave_size)])
        argv.extend(["--kv-order", args.kv_order])
        if getattr(args, "npu_reorder_selector", False):
            argv.append("--npu-reorder-selector")
        if getattr(args, "allow_identity_reorder", False):
            argv.append("--allow-identity-reorder")
    if not args.causal_fastpath:
        argv.append("--no-causal-fastpath")
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
    "1,4,4096,128",
    "2,4,8192,128",
    "2,8,16384,128",
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
                "--sparse-config", config,
                "--no-compare" if not args.compare else "",
            ]
            if args.enable_block_reorder:
                cmd.extend([
                    "--enable-block-reorder",
                    "--block-reorder-impl", args.block_reorder_impl,
                    "--block-reorder-mode", args.block_reorder_mode,
                    "--wave-size", str(args.wave_size),
                    "--kv-order", args.kv_order,
                ])
                if getattr(args, "npu_reorder_selector", False):
                    cmd.append("--npu-reorder-selector")
            if not args.causal_fastpath:
                cmd.append("--no-causal-fastpath")
            cmd = [c for c in cmd if c]
            label = f"  [{config}] {shape}"
            start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.warmup * args.repeat * 30 + 60,  # generous per test
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
    parser.add_argument("--target", choices=["both", "flex", "manual", "reorder"], default="both")
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
    parser.add_argument(
        "--save-output",
        default=None,
        help="Optional path to torch.save the primary output tensor for long-sequence correctness checks.",
    )
    parser.add_argument("--enable-block-reorder", action="store_true",
                        help="Enable block-level query reordering via spectral wave-overlap.")
    parser.add_argument(
        "--trim-outliers",
        dest="trim_outliers",
        action="store_true",
        default=True,
        help="Enable MAD-based outlier rejection on iteration times (requires repeat>=5).",
    )
    parser.add_argument(
        "--no-trim-outliers",
        dest="trim_outliers",
        action="store_false",
        help="Disable outlier rejection; use simple mean of all iteration times.",
    )
    parser.set_defaults(trim_outliers=True)
    parser.add_argument(
        "--causal-fastpath",
        dest="causal_fastpath",
        action="store_true",
        default=True,
        help="Enable causal dense fastpath for identity+causal (ROWS_GUARANTEED_SAFE + BLOCKS_ARE_CONTIGUOUS).",
    )
    parser.add_argument(
        "--no-causal-fastpath",
        dest="causal_fastpath",
        action="store_false",
        help="Disable causal fastpath to measure generic block-sparse template overhead.",
    )
    parser.set_defaults(causal_fastpath=True)
    parser.add_argument("--block-reorder-mode", default="wave_overlap",
                        choices=sorted(set(["identity", "wave_overlap"] + list(REORDER_REGISTRY.keys()) + list(LOCAL_REORDER_REGISTRY.keys()))),
                        help="Reorder mode (default: wave_overlap).")
    parser.add_argument(
        "--block-reorder-impl",
        choices=["internal", "external"],
        default="external",
        help="Reorder implementation. external is patent-style Q/metadata gather; internal is experimental and may segfault on NPU.",
    )
    parser.add_argument("--wave-size", type=int, default=132,
                        help="Wave partition size for reorder algorithms (default: 132).")
    parser.add_argument(
        "--kv-order",
        choices=["asc", "desc", "snake", "snake_inv", "boundary_dp", "edge_dp", "union_boundary_dp"],
        default="asc",
        help="KV column order for external FULL_KV reorder path.",
    )
    parser.add_argument(
        "--wave-single-tile-debug",
        action="store_true",
        help="Diagnostic mode: use wave grid but emit only one real Q-tile body per program.",
    )
    parser.add_argument(
        "--allow-identity-reorder",
        action="store_true",
        help="Run the external/internal reorder path even when the computed permutation is identity.",
    )
    parser.add_argument(
        "--npu-reorder-selector",
        action="store_true",
        help="Use a conservative NPU reorder selector instead of the raw CLI reorder mode/kv-order.",
    )
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
        parser.error("--mode profile-target requires --target flex, manual, or reorder")
    return args


if __name__ == "__main__":
    import torch._dynamo

    args = parse_args()
    torch._dynamo.config.suppress_errors = args.suppress_compile_errors

    if args.mode == "benchmark":
        score_mod, mask_mod, optimizations, extra_args = prepare_sparse_execution(args)
        result = run_shape_sweep(args, score_mod=score_mod, mask_mod=mask_mod,
                                 optimizations=optimizations, extra_args=extra_args)
        if args.save_output and result and "outputs" in result:
            output_key = None
            for candidate in ("flex_reorder", "flex", "manual"):
                if candidate in result["outputs"]:
                    output_key = candidate
                    break
            if output_key is None:
                raise RuntimeError("--save-output requested, but no output tensor was produced")
            output_path = Path(args.save_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result["outputs"][output_key].detach().cpu(), output_path)
            print(f"saved {output_key} output to {output_path}")
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
