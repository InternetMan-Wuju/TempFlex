# Flex Attention NPU 优化总结报告

> 日期: 2026-06-12 | 分支: `main`

## 1. 问题与方案

### 1.1 bishengir 限制

原有 `mask_mod(b,h,q_idx,kv_idx)` 在 kernel subgraph 中编译，触发多种 bishengir bug：

| 限制 | 现象 | 影响模式 |
|------|------|----------|
| `//` 整数除法 | `bool_to_bool_rintmode` MLIR 无法编译 | nested, strided, checkerboard, block_diagonal, dilated_window, uniform_doc, hybrid_sparse, multiscale_dilated |
| UB overflow | subgraph > 3 条件超出 UB 空间 (1.5MB) | global_local |
| `aten.index.Tensor` | tensor 下标索引不支持 | random_block_sparse (原始), alibi_causal |
| `torch.where` 链 | bishengir segfault | 含 4+ where 的 score_mod |

### 1.2 解决方案

```
旧: mask_mod(token_q, token_kv) → kernel subgraph → bishengir 编译
新: block_mask [1,1,MQ,NK] → FULL_KV_IDX → kernel 直接读取
```

**FULL_KV_IDX 的含义**：block mask 有两个并行通道。`KV_NUM_BLKS`/`KV_IDX` 是普通块（需要 mask_mod subgraph 逐 token 判断），`FULL_KV_NUM_BLKS`/`FULL_KV_IDX` 是全可见块（kernel 跳过 mask_mod，认为 block 内所有 token 互相可见）。我们把所有选中的 block 标为 FULL，普通块清零，kernel 就不走 mask_mod。

### 1.3 关键文件

| 文件 | 作用 |
|------|------|
| `pattern_to_block_mask.py` | 12 种 host 侧 block mask builder + FULL_KV 转换工具 |
| `Newest/.../flex_attention.py` | `pure_block_sparse_template` (subgraphs=[]) + PERM 模板 + lowering hook |
| `flex_attention_run_script.py` | benchmark 脚本 + 外部 Q reorder 流程 |
| `flex_attention_reorder.py` | perm 计算 (wave_overlap_reorder) |
| `sparse_masks.py` | 11 种 `*_bs` FULL_KV 配置 + 7 种原有配置 |

### 1.4 模式适配策略

| 模式类型 | 路径 | 代表 |
|----------|------|------|
| 简单规则模式 | causal fastpath（3 输入 dense，无 subgraph） | causal |
| block-aligned 模式 | FULL_KV 元数据 + generic 模板（block mask 完全定义 pattern） | block_diagonal, sliding_window, checkerboard 等 |
| block 内 partial mask | 第一版暂不支持（需 block 对齐或编译器修复） | uniform_doc 等 |

---

## 2. 正确性

> 测试条件: S=1024, B=4, H=8, D=128, bf16, warmup=10, repeat=10
> 正确性标准: flex vs manual 逐元素 allclose（rtol=0.02, atol=0.02, bf16）

| Pattern | Raw | Newest | Manual | allclose |
|---------|:---:|:------:|:------:|:--------:|
| `causal` | ✅ | ✅ | ✅ | 通过 |
| `random_block_sparse` | ✅ | ✅ | ✅ | 通过 |
| `block_diagonal_64_bs` | ✅ | ✅ | ✅ | 通过 |
| `sliding_window_128_bs` | ✅ | ✅ | ✅ | 通过 |
| `strided_bs` | ✅ | ✅ | ✅ | 通过 |
| `dilated_window_bs` | ✅ | ✅ | ✅ | 通过 |
| `nested_bs` | ✅ | ✅ | ✅ | 通过 |
| `hybrid_sparse_bs` | ✅ | ✅ | ✅ | 通过 |
| `checkerboard_64_bs` | ✅ | ✅ | ✅ | 通过 |
| `prefix_lm_bs` | ✅ | ✅ | ✅ | 通过 |

**全部 10 个模式正确性通过，fail_ratio = 0.0000%。**

---

## 3. 性能数据

> 测试条件: S=1024, B=4, H=8, D=128, bf16, warmup=10, repeat=10, --no-trim-outliers

### 3.1 S=1024: 3-Way 对比

| Pattern | Raw Flex | Newest Flex | Manual |
|---------|:--------:|:-----------:|:------:|
| `causal` | 4.31 ms | 2.65 ms | 0.59 ms |
| `random_block_sparse` ⚠️ | 4.28 ms | 2.65 ms | 0.59 ms |
| `block_diagonal_64_bs` | 0.56 ms | 0.54 ms | 0.80 ms |
| `sliding_window_128_bs` | 0.73 ms | 0.71 ms | 0.80 ms |
| `strided_bs` | 0.83 ms | 0.82 ms | 1.86 ms |
| `dilated_window_bs` | 0.85 ms | 0.84 ms | 0.80 ms |
| `nested_bs` | 0.94 ms | 0.93 ms | 0.81 ms |
| `hybrid_sparse_bs` | 1.04 ms | 1.02 ms | 0.79 ms |
| `checkerboard_64_bs` | 1.08 ms | 1.08 ms | 0.80 ms |
| `prefix_lm_bs` | 1.22 ms | 1.18 ms | 0.79 ms |

> ⚠️ `random_block_sparse` 的 4.28ms/2.65ms 是配置错误（见 §4.4），实际稀疏性能为 0.71ms。

### 3.2 S=8192 性能（Newest flex only）

| Pattern | S=1024 | S=8192 | vs Manual @ S=8192 (~58ms) |
|---------|:------:|:------:|:---------------------------:|
| `block_diagonal_64_bs` | 0.54 ms | 2.52 ms | **23x 快** |
| `sliding_window_128_bs` | 0.71 ms | 3.94 ms | **15x 快** |
| `dilated_window_bs` | 0.84 ms | 5.19 ms | **11x 快** |
| `nested_bs` | 0.93 ms | 15.7 ms | **3.7x 快** |
| `hybrid_sparse_bs` | 1.02 ms | 20.0 ms | **2.9x 快** |

> 以下模式 S=8192 编译/执行超时: `strided_bs`, `checkerboard_64_bs`, `prefix_lm_bs`（NPU 编译器在处理高密度 block mask 时的已知限制）。

---

## 4. 分析与结论

### 4.1 FULL_KV 模式在 Raw 和 Newest 上性能相同

**现象**：FULL_KV 模式 Raw vs Newest 差异在 ±3% 以内（测量噪声）。

**原因**：两者都走 generic 模板 + `HAS_FULL_BLOCKS=True` 路径。当所有 block 都是 FULL 时，generic 模板在**运行时**跳过 mask_mod subgraph：
```
if not IS_FULL_BLOCKS:
    // mask_mod subgraph  ← FULL block 时跳过
```
Newest 的 `pure_block_sparse_template`（`subgraphs=[]`）在**编译时**就确定没有 subgraph。但到了运行时，两者行为完全一致——都只做 identity score + 读 FULL_KV_IDX 跳 block。

**结论**：`pure_block_sparse_template` 的价值在于**绕开 bishengir 编译**（不编译 subgraph 就不会触发 `bool_to_bool_rintmode` 等问题），而不在于运行时加速。对已经能通过 mask_mod 编译的简单模式，没有额外收益。

### 4.2 Newest 在 dense 模式上比 Raw 快 1.6x

Newest 有 causal fastpath（3 输入 dense 模板），Raw 没有。对 `causal` 和 `random_block_sparse`（当前配置），Newest 2.65ms vs Raw 4.31ms。

### 4.3 Flex vs Manual 的密度拐点 ≈ 35%

| Density | Flex vs Manual | 代表模式 |
|:-------:|:--------------|----------|
| < 30% | Flex 更快（跳过无效 KV blocks） | block_diagonal (12.5%): **1.5x 快** |
| ~35% | 临界点 | strided (31.3%): **2.3x 快**；dilated_window (32.8%): 1.1x 慢 |
| > 40% | Manual 更快（block-sparse metadata 开销 > 节省） | prefix_lm (56.3%): 1.5x 慢 |

### 4.4 `random_block_sparse` 配置错误

**问题**：当前配置设了 `ROWS_GUARANTEED_SAFE: True`，触发了 causal fastpath（3 输入密集模板），**完全忽略** kv_indices block mask。Flex 和 Manual 都在算 100% dense causal 注意力，所以性能和 `causal` 模式一样（~4.3ms Raw / ~2.6ms Newest）。

**修正**：禁用 fastpath，走真正的 block-sparse（23% density）：

| 配置 | Flex | Manual |
|------|------|--------|
| 错误 (dense fastpath) | 2.65 ms | 0.59 ms |
| 正确 (sparse, 23%) | **0.71 ms** | 0.82 ms |

**教训**：`ROWS_GUARANTEED_SAFE` + `BLOCKS_ARE_CONTIGUOUS` 会绕过 block mask，只适用于真正的 dense causal。不能用于任何稀疏配置。

### 4.5 已接入 new-vllm-omni 参考代码

外部 Q reorder 流程（`flex_attention_run_script.py`）遵循 `sparse-attn-source` 中 `flash_attn.py` L865-940 的实现：
1. Host 侧构建 block mask → CPU `wave_overlap_reorder` 计算 perm
2. `torch.gather` 重排 Q blocks + kv_indices/kv_num_blocks rows
3. Route ALL blocks as FULL + `PURE_BLOCK_SPARSE` kernel option
4. `torch.gather` + inv_perm 逆重排 output

---

## 5. Reorder 状态

| 方案 | 可用规模 | 状态 |
|------|----------|------|
| 外部 Q reorder + pure block-sparse | 所有规模 | ✅ 正确性已验证 |
| Kernel-internal PERM reorder | S≤2048 | ⚠️ S≥4096 编译器 data-dependency 死锁 |

---

## 6. 部署

```bash
bash Newest/apply_newest.sh    # 部署开发版
bash raw_flex/apply_raw.sh     # 回退原始版
```

```bash
# FULL_KV 模式（推荐）
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs --seq-len 8192 --target both

# 开启 reorder
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs --enable-block-reorder
```

---

## 7. 下一步

| 优先级 | 任务 |
|--------|------|
| P0 | 上报 NPU 编译器 data-dependency 死锁 bug（影响 PERM reorder S≥4096 + PURE_BLOCK_SPARSE_CAUSAL） |
| P1 | 高密度模式 S=8192 编译超时修复（影响 checkerboard/strided/prefix_lm） |
| P2 | `random_block_sparse` 创建真正的稀疏 benchmark 配置（禁用 fastpath） |
| P3 | Partial block 支持（uniform_doc 等 document-boundary 模式） |
| P4 | Per-head block mask |
