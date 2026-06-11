---
name: Flex_attn_wave_reorder_plan
description: Wave-based Kernel Reorder 实现计划 — 架构设计、分步实施、测试策略
metadata:
  skill: true
  updated: 2026-06-11
---

# Wave-based Kernel Reorder 实现计划

## 1. 问题定位

### 1.1 当前 Reorder 反而更慢

```
实测数据（random_block_sparse）：
S=2048: no reorder 1.26ms → reorder 2.98ms (慢 2.4x)
S=8192: no reorder 8.24ms → reorder 34.9ms (慢 4.2x)
```

**根因**：每个 Q block = 一个独立 program。原始顺序下 Q 内存访问是连续的（coalesced DRAM），reorder 后 program N 加载 Q block[perm[N]]，访问变为散乱（scattered），DRAM 带宽效率崩溃。

```
原始: program[0]→Q[0:64]  program[1]→Q[64:128]  ← 连续 DRAM 访问
重排: program[0]→Q[960:1024]  program[1]→Q[896:960]  ← 散乱 DRAM 访问
```

### 1.2 为什么 GPU vllm-omni 有效

GPU 上一个 thread block 处理多个 Q blocks，Q 数据在 block 内连续加载，KV 数据在 L2 cache 中复用。

```
GPU (vllm-omni):  NPU (当前):
  wave[0..7]        program[0]
  ↓                 ↓
  8 Q blocks        1 Q block
  共享 KV cache     独立加载 KV
  ↓                 ↓
  reorder 有效       reorder 无效
```

## 2. 方案设计

### 2.1 核心思路

**让每个 program 处理 WAVE_SIZE 个 Q blocks，在 program 内部按 reorder permutation 排序。**

```
当前：program[i] → Q[perm[i]] → KV blocks → compute → output[perm[i]]
Wave：program[w] → Q[perm[w*K..w*K+K-1]] → KV blocks → compute ×K → output[...]
```

### 2.2 架构变化

```
┌─────────────────────────────────────────────────────┐
│                   flex_attention 调用                 │
├─────────────────────────────────────────────────────┤
│  1. compute_and_set_pending_perm(kv_indices)        │
│     → 计算 mask-level perm，存入 side-channel        │
│  2. flex_attention(q, k, v, block_mask,             │
│       kernel_options={                              │
│         "WAVE_SIZE": K,                             │
│         "ENABLE_REORDER": True,                     │
│       })                                            │
│  3. lowering 从 side-channel 取出 perm              │
│  4. 生成 wave-variant kernel                        │
│  5. kernel 执行                                     │
└─────────────────────────────────────────────────────┘
```

### 2.3 Kernel 模板变化

```
BEFORE（当前）:                        AFTER（Wave）:
─────────────────────────            ─────────────────────────
q_start = program_id(0)              wave_id = program_id(0)
                                      wave_start = wave_id * WAVE_SIZE
if ENABLE_REORDER:                    for w in range(WAVE_SIZE):
    src_block = PERM[q_start]             q_start = wave_start + w
else:                                     if ENABLE_REORDER:
    src_block = q_start                       src_block = PERM[q_start]
                                          else:
// Q block 处理                           src_block = q_start
offs_m = src_block * BLOCK_M
Q_block_ptr(offsets=(src_block*BM,0))   // Q block 处理（同左）
forward_inner(...)                      offs_m = src_block * BLOCK_M
store_output(...)                       Q_block_ptr(offsets=(src_block*BM,0))
                                        forward_inner(...)
                                        store_output(...)
                                      // end of wave loop
```

### 2.4 Grid 调整

```
BEFORE: grid[0] = ceil(Q_LEN / BLOCK_M)
AFTER:  grid[0] = ceil(Q_LEN / (BLOCK_M * WAVE_SIZE))
```

### 2.5 PERM 格式

PERM 是 mask-level permutation（大小 = ceil(Q_LEN / SPARSE_Q_BLOCK_SIZE)）。
Kernel 内部通过 `SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M` 展开到 triton-level。

```
q_start 是 triton-level 索引（0..n_triton_blocks-1）
src_mask_block = PERM[q_start // SPARSE_Q_MULTIPLE]   // mask-level lookup
src_block = src_mask_block * SPARSE_Q_MULTIPLE + (q_start % SPARSE_Q_MULTIPLE)
```

### 2.6 正确性保证

- **Q tensor 不重排**：score_mod 看到正确的 token_q
- **kv_indices 不重排**：每个 Q block 使用原始 mask
- **输出自动在原位**：`store_output` 写入 `src_block * BLOCK_M` 位置
- **WAVE_SIZE=1** 行为完全等同于当前 kernel

## 3. 分步实施

### Step 1：生成 Wave Kernel 模板（不修改原模板）

**方法**：用 Python 字符串操作程序化生成 wave variant，而非手动编辑。

```python
def generate_wave_template(base_source, wave_size):
    """
    输入：原始 compute_flex_attention 字符串
    输出：添加了 wave loop 的变体字符串
    """
    # 1. 找到 Q block 处理段的起止位置
    #    start: "initialize pointer to m and l"
    #    end:   OUTPUT_LOGSUMEXP block 结束
    #
    # 2. 在 start 前插入：
    #    wave_start = wave_id * WAVE_SIZE
    #    for wave_w in range(WAVE_SIZE):
    #        q_start = wave_start + wave_w
    #        if ENABLE_REORDER:
    #            src_block = PERM[...]
    #        else:
    #            src_block = q_start
    #
    # 3. 将 Q block 处理段每行加 4 空格缩进
    # 4. 在 end 后关闭循环
    #
    # 5. 将 q_start = program_id(0) 改为 wave_id = program_id(0)
    # 6. 移除原始 PERM 块（已移入 loop 内）
    #
    return modified_source
```

**关键**：用 `assert` 验证生成代码的关键 marker 都存在。

### Step 2：添加 WAVE_SIZE 到 kernel_options

```python
# flex_attention.py lowering 函数中
kernel_options.setdefault("WAVE_SIZE", 1)
```

### Step 3：Override Grid 函数

```python
def _flex_attention_grid_with_wave(*args, **kwargs):
    grid = list(flex_attention_grid(*args, **kwargs))
    wave_size = kwargs.get('WAVE_SIZE', 1)
    if wave_size > 1:
        grid[0] = max(1, (grid[0] + wave_size - 1) // wave_size)
    return tuple(grid)
```

### Step 4：注册两个 Variant

```python
# variant 1: WAVE_SIZE=1 (原始，每次都要)
flex_attention_template.maybe_append_choice(...)

# variant 2: WAVE_SIZE>1 (如果 kernel_options 指定)
if kernel_options.get('WAVE_SIZE', 1) > 1:
    wave_template = TritonTemplate(
        name="flex_attention_wave",
        grid=_flex_attention_grid_with_wave,
        source=generate_wave_template(compute_flex_attention),
    )
    wave_template.maybe_append_choice(...)
```

### Step 5：向后兼容

- `WAVE_SIZE=1` 使用原始 template（不变）
- `WAVE_SIZE>1` 使用 wave template
- Backward kernel 保持 `WAVE_SIZE=1`（暂不支持）
- 其他所有参数（ENABLE_REORDER, PERM, BLOCK_M/N 等）保持一致

## 4. 文件变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `flex_attention.py` | +80 行 | generate_wave_template, grid override, wave template 注册 |
| `flex_attention_reorder.py` | 不变 | compute_and_set_pending_perm 已就绪 |
| `flex_attention_run_script.py` | +5 行 | 支持 --wave-size 参数 |
| `sparse_masks.py` | 不变 | 模式配置已就绪 |

## 5. 测试计划

### 5.1 正确性测试

```bash
# WAVE_SIZE=1, ENABLE_REORDER=True → 应与无 wave 版本完全一致
python3 flex_attention_run_script.py --shape 1,4,512,64 \
  --enable-block-reorder --no-compare

# WAVE_SIZE=2, ENABLE_REORDER=True → 非 identity perm，应通过 allclose
python3 flex_attention_run_script.py --shape 1,4,512,64 \
  --enable-block-reorder --sparse-config sliding_window_64 \
  --wave-size 2

# 所有 7 个支持的模式 + 不同 WAVE_SIZE
for cfg in causal sliding_window_64 global_local random_block_sparse; do
  for ws in 1 2 4; do
    python3 flex_attention_run_script.py \
      --shape 1,4,512,64 --sparse-config $cfg \
      --enable-block-reorder --wave-size $ws --no-compare
  done
done
```

### 5.2 性能测试

```bash
# 对比不同 WAVE_SIZE
for ws in 1 2 4 8; do
  echo "WAVE_SIZE=$ws"
  python3 flex_attention_run_script.py \
    --shape 1,4,4096,64 \
    --sparse-config random_block_sparse \
    --target flex --no-compare \
    --enable-block-reorder --wave-size $ws
done
```

### 5.3 msprof 验证

```bash
# 采集 wave kernel 的 Memory metrics
python3 flex_attention_run_script.py --mode msprof \
  --shape 1,4,4096,64 \
  --sparse-config random_block_sparse \
  --target flex \
  --enable-block-reorder --wave-size 4 \
  --msprof-aic-metrics "Memory"
```

### 5.4 通过标准

- [ ] WAVE_SIZE=1 输出与原始 kernel 完全一致（allclose）
- [ ] WAVE_SIZE>1 + ENABLE_REORDER 输出与 manual baseline 一致（allclose）
- [ ] WAVE_SIZE>1 比 WAVE_SIZE=1 **快**（性能提升）
- [ ] hit_rate < 1.0 的模式在 wave 下有更大收益

## 6. 预期效果

| 指标 | 当前 | Wave (预期) |
|------|------|------------|
| S=2048, reorder | 2.98ms (2.4x slower) | < 1.5ms (接近 baseline 1.26ms) |
| S=8192, reorder | 34.9ms (4.2x slower) | < 12ms (缩小差距) |
| Program count | ceil(S/BLOCK_M) | ceil(S/(BLOCK_M*WAVE_SIZE)) |
| Q DRAM 访问 | scattered | sequential（每 wave 连续加载） |
| KV cache 复用 | 无 | 同 wave 内复用（future） |

## 7. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 模板生成 indentation 错误 | 高 | assert 验证，git diff 检查，只改目标行 |
| BLOCK_M*WAVE_SIZE 超过 SPARSE_Q_BLOCK_SIZE | 中 | 限制 WAVE_SIZE ≤ SPARSE_Q_BLOCK_SIZE // BLOCK_M |
| 每个 program SRAM 用量增加 | 中 | 从 WAVE_SIZE=2 开始，逐步增加 |
| bishengir 编译大 kernel 失败 | 中 | 先用小 S 测试，逐步放大 |
| backward 不支持 | 低 | backward 禁用 wave（WAVE_SIZE=1） |

## 8. 后续迭代

- **Phase 2**: KV cache 复用 — 同 wave 内 Q blocks 共享已加载的 KV blocks
- **Phase 3**: Auction reorder 算法 — 从 sparse-attn-source 移植更优的 reorder 算法
- **Phase 4**: backward 支持 — 逆向传播时也使用 wave
