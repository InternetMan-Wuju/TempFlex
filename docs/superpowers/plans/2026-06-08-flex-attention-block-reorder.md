# NPU Flex Attention Block Reorder 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 NPU flex attention 基准测试中实现 block reorder（wave_overlap 算法），验证 hit rate 提升和性能加速效果。

**Architecture:** 新建 `flex_attention_reorder.py` 模块（纯 PyTorch），包含 `wave_overlap_reorder` 算法和 `reorder_flex_forward` 管道；修改 `flex_attention_run_script.py` 恢复 `--enable-block-reorder` 支持；修改 `apply_newest.sh` 部署新模块。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, Triton (NPU Inductor)

---

### Task 1: 新建 `flex_attention_reorder.py` — 核心函数

**Files:**
- Create: `Newest/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py`

- [ ] **Step 1: 创建模块头和 `compute_block_hit_rate`**

```python
"""Block reorder utilities for NPU Flex Attention.

Provides algorithms to reorder KV blocks accessed by each Q block row,
improving L2 cache locality and attention speed.
"""

import math
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


def compute_block_hit_rate(kv_indices, kv_num_blocks):
    """Compute the cache hit rate of KV block access pattern.

    A "hit" occurs when consecutive KV blocks accessed by a Q row
    are also consecutive in memory (index difference == 1).

    Args:
        kv_indices: Tensor[B, H, n_blocks, max_blocks] — per-row KV block indices
        kv_num_blocks: Tensor[B, H, n_blocks] — number of valid blocks per row

    Returns:
        float: hit rate in [0, 1]
    """
    B, H, n_blocks, max_blocks = kv_indices.shape
    total_transitions = 0
    total_hits = 0

    for b in range(B):
        for h in range(H):
            for row in range(n_blocks):
                n = int(kv_num_blocks[b, h, row].item())
                if n <= 1:
                    continue
                indices = kv_indices[b, h, row, :n]
                diffs = indices[1:] - indices[:-1]
                hits = (diffs == 1).sum().item()
                total_hits += hits
                total_transitions += (n - 1)

    if total_transitions == 0:
        return 0.0
    return total_hits / total_transitions
```

- [ ] **Step 2: 实现 `rebuild_block_mask`**

```python
def rebuild_block_mask(kv_num_blocks, kv_indices, n_blocks_q, n_blocks_kv, device=None):
    """Convert sparse block mask representation to dense mask.

    Args:
        kv_num_blocks: Tensor[B, H, n_blocks] — valid block count per row
        kv_indices: Tensor[B, H, n_blocks, max_blocks] — block indices per row
        n_blocks_q: int — total number of Q blocks
        n_blocks_kv: int — total number of KV blocks
        device: optional device override

    Returns:
        mask_float: Tensor[B, n_blocks_q, n_blocks_kv] — 0/1 float mask
    """
    if device is None:
        device = kv_indices.device
    B, H, n_blocks, max_blocks = kv_indices.shape
    # Collapse batch and head for reorder (treat each head independently)
    mask = torch.zeros(B, n_blocks_q, n_blocks_kv, device=device, dtype=torch.float32)
    
    for b in range(B):
        for row in range(n_blocks):
            n = int(kv_num_blocks[b, 0, row].item())
            if n == 0:
                continue
            idx = kv_indices[b, 0, row, :n]
            mask[b, row, idx.long()] = 1.0

    return mask
```

- [ ] **Step 3: 实现核心算法 `wave_overlap_reorder`**

```python
def wave_overlap_reorder(mask_float, wave_size=132, n_iter=10):
    """Wave-Overlap Reorder: 1D Spectral Sorting via Power Iteration.

    Ported from GPU vllm-omni (greedy_reorder_cuda.py:2441).

    Uses power iteration to find the top-1 left singular vector (Fiedler
    vector approximation) of the block mask, then sorts rows by their
    projection onto this vector. Rows with similar KV access patterns
    are clustered together, improving cache locality.

    Args:
        mask_float: Tensor[..., MB, NB] — dense block mask (any leading dims)
        wave_size: int — wave partition size (default 132)
        n_iter: int — power iteration steps (default 10)

    Returns:
        perm: Tensor[..., MB] — row permutation indices
    """
    *leading, MB, NB = mask_float.shape
    H = 1 if not leading else math.prod(leading)
    mask_2d = mask_float.reshape(H, MB, NB)
    device = mask_float.device

    # Power iteration for top-1 left singular vector
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 1, device=device, generator=gen)

    for _ in range(n_iter):
        U = torch.bmm(mask_2d, V)  # (H, MB, 1)
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V = torch.bmm(mask_2d.transpose(1, 2), U)  # (H, NB, 1)
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Project rows onto the singular vector and sort
    proj = torch.bmm(mask_2d, V).squeeze(-1)  # (H, MB)
    perm = torch.argsort(proj, dim=1)  # (H, MB)

    # Reshape back to original leading dims
    if leading:
        perm = perm.reshape(*leading, MB)
    return perm
```

- [ ] **Step 4: 实现 `ReorderInfo` 数据类**

```python
@dataclass
class ReorderInfo:
    """Result of a block reorder operation."""
    perm: torch.Tensor                    # (H, MB) row permutation
    kv_indices: torch.Tensor              # reordered kv_indices
    kv_num_blocks: torch.Tensor           # reordered kv_num_blocks
    full_kv_indices: Optional[torch.Tensor] = None
    full_kv_num_blocks: Optional[torch.Tensor] = None
    query: Optional[torch.Tensor] = None  # reordered query (None if no Q reorder)
    baseline_hit_rate: float = 0.0
    reordered_hit_rate: float = 0.0
    reorder_mode: str = "wave_overlap"
```

- [ ] **Step 5: 实现主入口 `reorder_flex_forward`**

```python
def reorder_flex_forward(
    q, k, v,
    kv_num_blocks, kv_indices,
    full_kv_num_blocks, full_kv_indices,
    BLOCK_SIZE_M, BLOCK_SIZE_N,
    mode="wave_overlap",
    wave_size=132,
    sorted_kv=True,
    verbose=False,
):
    """Apply block reorder to flex attention inputs.

    This function takes the existing block mask, runs the selected reorder
    algorithm, and returns reordered indices + query.

    Args:
        q, k, v: QKV tensors
        kv_num_blocks, kv_indices: original block mask
        full_kv_num_blocks, full_kv_indices: full block mask (optional)
        BLOCK_SIZE_M, BLOCK_SIZE_N: block sizes
        mode: reorder algorithm name (default "wave_overlap")
        wave_size: wave partition size for reorder
        sorted_kv: whether to sort KV indices within each row
        verbose: print debug info

    Returns:
        ReorderInfo with reordered data
    """
    B, H_q, seq_len_q, head_dim = q.shape
    _, H_kv, seq_len_kv, _ = k.shape

    n_blocks_q = (seq_len_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_blocks_kv = (seq_len_kv + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Compute baseline hit rate
    device = kv_indices.device
    baseline_hit = compute_block_hit_rate(kv_indices, kv_num_blocks)
    if verbose:
        print(f"[reorder] Baseline hit rate: {baseline_hit:.4f}")

    # Build dense mask from kv_indices
    n_blocks = min(n_blocks_q, kv_indices.shape[2])
    mask_float = rebuild_block_mask(
        kv_num_blocks, kv_indices, n_blocks, n_blocks_kv, device=device
    )

    # Run reorder algorithm
    if mode == "wave_overlap":
        perm = wave_overlap_reorder(mask_float, wave_size=wave_size)
    elif callable(mode):
        perm = mode(mask_float, wave_size=wave_size)
    else:
        raise ValueError(f"Unknown reorder mode: {mode}")

    # Reorder kv_indices and kv_num_blocks rows by perm
    reordered_kv_indices = torch.zeros_like(kv_indices)
    reordered_kv_num_blocks = torch.zeros_like(kv_num_blocks)

    for row in range(n_blocks):
        src_row = perm[0, row].item()
        reordered_kv_num_blocks[..., row, :] = kv_num_blocks[..., src_row, :]
        reordered_kv_indices[..., row, :] = kv_indices[..., src_row, :]

    # Reorder full blocks if present
    reordered_full_kv_indices = None
    reordered_full_kv_num_blocks = None
    if full_kv_num_blocks is not None and full_kv_indices is not None:
        reordered_full_kv_num_blocks = torch.zeros_like(full_kv_num_blocks)
        reordered_full_kv_indices = torch.zeros_like(full_kv_indices)
        n_full_rows = min(full_kv_num_blocks.shape[2], perm.shape[1])
        for row in range(n_full_rows):
            src_row = perm[0, row].item()
            reordered_full_kv_num_blocks[..., row, :] = full_kv_num_blocks[..., src_row, :]
            reordered_full_kv_indices[..., row, :] = full_kv_indices[..., src_row, :]

    # Reorder Q along the sequence dimension
    # We need to reorder Q's rows to match the reordered block access pattern
    reordered_q = None
    if True:  # Always reorder Q for now
        # Permute Q blocks (row blocks in sequence)
        q_blocks = q.reshape(B, H_q, n_blocks_q, BLOCK_SIZE_M, head_dim)
        perm_expanded = perm[0, :n_blocks_q].long()
        reordered_q_blocks = q_blocks[:, :, perm_expanded]
        reordered_q = reordered_q_blocks.reshape(B, H_q, n_blocks_q * BLOCK_SIZE_M, head_dim)
        # Truncate to original sequence length
        reordered_q = reordered_q[:, :, :seq_len_q, :]

    # Compute reordered hit rate
    reordered_hit = compute_block_hit_rate(reordered_kv_indices, reordered_kv_num_blocks)
    if verbose:
        print(f"[reorder] Reordered hit rate: {reordered_hit:.4f} "
              f"(delta: {reordered_hit - baseline_hit:+.4f})")

    return ReorderInfo(
        perm=perm,
        kv_indices=reordered_kv_indices,
        kv_num_blocks=reordered_kv_num_blocks,
        full_kv_indices=reordered_full_kv_indices,
        full_kv_num_blocks=reordered_full_kv_num_blocks,
        query=reordered_q,
        baseline_hit_rate=baseline_hit,
        reordered_hit_rate=reordered_hit,
        reorder_mode=mode,
    ), baseline_hit, reordered_hit
```

- [ ] **Step 6: 实现 `REORDER_REGISTRY`**

```python
REORDER_REGISTRY = {
    "wave_overlap": wave_overlap_reorder,
}
```

- [ ] **Step 7: 验证模块可导入**

Run:
```bash
PYTHONPATH=/wyh/code/TempFlex/Newest/site-packages:$PYTHONPATH \
  /usr/local/python3.11.14/bin/python3.11 \
  -c "import sys; sys.path.insert(0, 'Newest/site-packages'); from torch_npu._inductor.kernel.flex_attention_reorder import *; print('OK')"
```
Expected: `OK`

---

### Task 2: 修改 `flex_attention_run_script.py` — 恢复 reorder 支持

**Files:**
- Create: `.gitignore` (add Newest/site-packages entry)
- Modify: `flex_attention_run_script.py`

- [ ] **Step 1: 在文件头部加入 reorder 模块的 import**

```python
from torch_npu._inductor.kernel.flex_attention_reorder import (
    reorder_flex_forward,
    compute_block_hit_rate,
    REORDER_REGISTRY,
)
```
放在第 28-34 行的位置（替代之前的 import 块）。

- [ ] **Step 2: 在 `run_benchmark()` 函数中恢复 reorder 分支**

在 `run_benchmark` 中（约第 774 行），替换原有的 `# ── Reorder variant` 注释块为：

```python
    # ── Reorder variant: only when --enable-block-reorder and NPU ──
    reorder_hit_rate = None
    if args.enable_block_reorder and device_is_npu(args.device):
        block_mask_device = "cpu" if device_is_npu(args.device) else args.device
        bm = create_block_mask(
            mask_mod, 1, 1, args.seq_len, args.seq_len,
            device=block_mask_device,
        ).to(args.device)

        baseline_hit = compute_block_hit_rate(bm.kv_indices, bm.kv_num_blocks)

        # Dispatch to selected reorder mode
        info, baseline_hit_rate, reordered_hit = reorder_flex_forward(
            q, k, v,
            bm.kv_num_blocks, bm.kv_indices,
            bm.full_kv_num_blocks, bm.full_kv_indices,
            bm.BLOCK_SIZE[0], bm.BLOCK_SIZE[1],
            mode=args.block_reorder_mode,
            wave_size=args.wave_size,
            sorted_kv=True,
            verbose=True,
        )

        reorder_hit_rate = (baseline_hit, reordered_hit)

        bm_reordered = BlockMask.from_kv_blocks(
            kv_num_blocks=info.kv_num_blocks,
            kv_indices=info.kv_indices,
            full_kv_num_blocks=info.full_kv_num_blocks,
            full_kv_indices=info.full_kv_indices,
            BLOCK_SIZE=bm.BLOCK_SIZE,
            mask_mod=bm.mask_mod,
        )

        # Use reordered Q + reordered block mask
        if info.query is not None:
            q_reordered = info.query
        else:
            q_reordered = q

        reorder_runner = make_flex_runner(
            q_reordered, k, v, score_mod, mask_mod, args,
            kernel_options_extra=merged_kernel_options,
            block_mask=bm_reordered,
        )

        outputs["flex_reorder"], timings["flex_reorder"] = time_runner(
            f"Flex+{args.block_reorder_mode}", reorder_runner, args)
```

- [ ] **Step 3: 在 detailed_compare 中新增 reorder 精度对比**

在原有的 reorder vs baseline 比较代码块后（约第 882 行），确保输出中包含 hit rate 信息：

```python
            if reorder_hit_rate is not None:
                print(f"  Hit rate: {reorder_hit_rate[0]:.4f} → {reorder_hit_rate[1]:.4f} "
                      f"(delta: {reorder_hit_rate[1] - reorder_hit_rate[0]:+.4f})")
```

- [ ] **Step 4: 验证脚本可运行**

Run:
```bash
cd /wyh/code/TempFlex
/usr/local/python3.11.14/bin/python3.11 flex_attention_run_script.py --help | grep -i reorder
```
Expected: 输出包含 `--enable-block-reorder`、`--block-reorder-mode`、`--wave-size`

---

### Task 3: 修改 `apply_newest.sh` — 部署新模块

**Files:**
- Modify: `Newest/apply_newest.sh`

- [ ] **Step 1: 加入 reorder.py 的部署命令**

在 `flex_attention.py` 的 copy_one 之后追加：

```bash
echo "[INFO] applying flex attention reorder module ..."

copy_one "${SCRIPT_DIR}/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py" \
         "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py"

echo "[OK] flex attention reorder module replaced"
```

- [ ] **Step 2: 验证部署脚本执行**

Run:
```bash
cd /wyh/code/TempFlex
bash Newest/apply_newest.sh 2>&1 | grep -i "reorder"
```
Expected: `[INSTALL] ...flex_attention_reorder.py -> ...`

---

### Task 4: 跑 causal-only 验证管道通

- [ ] **Step 1: 安装新模块**

```bash
cd /wyh/code/TempFlex
bash Newest/apply_newest.sh 2>&1
```

- [ ] **Step 2: 测试 causal 单条命令**

```bash
/usr/local/python3.11.14/bin/python3.11 \
  flex_attention_run_script.py \
  --shape 4,8,2048,128 \
  --sparse-config causal \
  --enable-block-reorder \
  --block-reorder-mode wave_overlap \
  --warmup 1 --repeat 1
```
Expected: 正常输出 flex 和 reorder 的耗时和 hit rate。

- [ ] **Step 3: 用 sweep 脚本跑 --only-causal**

```bash
/usr/local/python3.11.14/bin/python3.11 \
  run_sparse_sweep.py \
  --only-causal \
  --enable-block-reorder \
  --reorder-mode wave_overlap
```
Expected: 每个 shape 的 reorder 列有数据，hit rate 显示在输出中。

---

### Task 5: 验证非 causal 配置

- [ ] **Step 1: 测试一个非 causal 配置**

```bash
/usr/local/python3.11.14/bin/python3.11 \
  flex_attention_run_script.py \
  --shape 4,8,2048,128 \
  --sparse-config sliding_window_64 \
  --enable-block-reorder \
  --block-reorder-mode wave_overlap \
  --warmup 1 --repeat 1
```

- [ ] **Step 2: 记录 causal vs 非 causal 的 hit rate 差异**

在两者上都收集 baseline hit rate 和 reordered hit rate，比较改善幅度。