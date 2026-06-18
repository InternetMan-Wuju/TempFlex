# Flex Attention NPU 优化总结报告

> 日期: 2026-06-18  
> 设备: Ascend NPU 60GB  
> 精度: bfloat16  
> 主要对比: Raw flex / Newest without reorder / Newest with external reorder / Manual reference

## 1. 核心结论

1. **Causal 不需要 reorder**。Newest 的 causal dense fastpath 已经能带来稳定收益，短中序列约 `1.1x-1.37x`，长序列仍有 `1.04x-1.09x`。
2. **12 种稀疏模式能跑通的关键不是 row reorder，而是 FULL_KV / PURE_BLOCK_SPARSE 路径**。它把复杂 `mask_mod` subgraph 移到 host 侧生成 metadata，绕开 bishengir 对 `//`、`%`、`torch.where`、tensor index 等 lowering 的不稳定。
3. **16K 上 row/KV reorder 本身确实有用，但集中在部分模式**。最新 optimized fair A/B（identity external vs optimized reorder external）下，`strided_bs`、`prefix_lm_bs`、`nested_bs`、`hybrid_sparse_bs` 分别为 `1.0363x / 1.0313x / 1.0290x / 1.0195x`；`checkerboard_64_bs`、`band_global_bs` 仍会退化。
4. **32K/80K 下普通 no-reorder vs reorder/external 不是公平 speedup**。Raw 和 Newest without reorder 在非 causal FULL_KV sparse 长序列下无法稳定完成；因此不能用普通路径的 `CRASH/ERR(-9)` 去计算 reorder 加速。
5. **公平 A/B 口径是 identity external vs optimized reorder external**。两边都预先重排/拷贝 Q 和 sparse metadata，只改变 row/KV order。本轮 32K 表中 `strided_bs / nested_bs / hybrid_sparse_bs / prefix_lm_bs` 均达到 `>=1.01x` 且 allclose 通过。
6. **80K reorder 仍不适合默认开启**。补测后 80K crash 行多数已补齐：`nested_bs` 小幅 `1.0047x`，`strided_bs` 基本持平，`dilated_window_bs` 略慢；`hybrid_sparse_bs` 的 exact-path 仍 crash，但 fallback `wave_overlap + snake_inv` 可跑通并有 `1.0036x` 小收益。当前 selector 仍不默认开启 80K reorder。`BLOCK_N=128` probe 显示 FULL_KV template 有更大性能空间，但 reorder 不稳定，因此下一步应优化默认 `BLOCK_N=64` 下的 `get_offset_for_next_block()` / FULL_KV traversal。

## 2. 实现口径

### 2.1 四条测试路径

| 路径 | 含义 | 主要用途 |
|---|---|---|
| Raw flex | 部署 `raw_flex` 后运行 `--target flex` | 原始实现基线 |
| Newest without reorder | 部署 `Newest` 后运行 `--target flex` | 新版普通 flex 路径 |
| Newest with reorder | `--target reorder --enable-block-reorder --block-reorder-impl external`，按 optimized fair selector 选择 reorder mode / KV order | external reorder 路径 |
| Manual | `--target manual --no-compare` | Python/torch 参考实现或 dense reference |

### 2.2 公平性说明

`Newest with reorder/external` 不是只“换执行顺序”。它会在 kernel 外提前重排 Q 和 sparse metadata，让 flex kernel 继续走更稳定的单 tile / pure block-sparse 模板。  

因此报告里把两类问题分开：

- **可运行性对比**：Raw / Newest without reorder 是否能跑完。
- **公平 reorder A/B**：identity external vs optimized reorder external。只有这个口径能比较 reorder 本身收益或退化。

## 3. 稀疏模式支持: Subgraph -> Metadata

原始实现通过 `mask_mod(b, h, q_idx, kv_idx)` 在 kernel 内动态计算稀疏 pattern。bishengir 对这类 subgraph 不稳定：

| 操作 | 风险 | 受影响模式 |
|---|---|---|
| `//` | `bool_to_bool_rintmode` crash | nested, strided, checkerboard, dilated, hybrid 等 |
| `%` | `aten.remainder.Scalar` lowering 失败 | 多数规则稀疏 |
| `torch.where` 链 | segfault / UB overflow | global_local, band_global |
| tensor index | NPU 不支持 tensor 下标索引 | random/alibi 类模式 |

当前方案把复杂逻辑全部移到 host 侧：

1. `pattern_to_block_mask.py` 生成 block-level bool mask。
2. `block_mask_to_full_kv()` 转换为 `FULL_KV_NUM_BLKS / FULL_KV_IDX` metadata。
3. PURE_BLOCK_SPARSE kernel 删除 `mask_mod` / `score_mod` subgraph，只按 metadata 遍历有效 KV blocks。

重要口径：**FULL_KV 不等于 dense**。FULL_KV 表示“有效 block 走 full-block metadata 路径”，不是把所有 KV block 都标记有效。每行实际加载的 block 数仍由 `full_kv_num_blocks` 决定。

## 4. Causal Fastpath

Newest 新增 causal dense fastpath，使用 3 输入专用模板，无 subgraph。Causal 本身自然局部性强，默认不启用 reorder。

| Shape | Raw | Newest | Speedup |
|---|---:|---:|---:|
| 1,2,1024 | 0.60 ms | 0.46 ms | 1.29x |
| 1,2,2048 | 0.99 ms | 0.72 ms | 1.37x |
| 1,2,4096 | 1.92 ms | 1.49 ms | 1.29x |
| 1,2,8192 | 4.79 ms | 3.93 ms | 1.22x |
| 2,4,8192 | 17.5 ms | 14.2 ms | 1.23x |
| 4,8,8192 | 67.9 ms | 55.4 ms | 1.22x |
| 1,2,16384 | 14.4 ms | 13.0 ms | 1.11x |
| 4,8,16384 | 219 ms | 195 ms | 1.12x |
| 2,4,32768 | 196 ms | 184 ms | 1.07x |
| 1,2,49152 | 108 ms | 103 ms | 1.04x |
| 1,2,65536 | 196 ms | 180 ms | 1.09x |

## 5. 1K / 4K / 16K 总表

测试配置：`B=1, H=2, D=128`。Flex / Reorder 使用 `warmup=3, repeat=5`；Manual 使用 `warmup=1, repeat=3`。

### 5.1 图表

![S=1024 runtime by sparse type](reports/raw_newest_reorder_manual_s1024.svg)

![S=4096 runtime by sparse type](reports/raw_newest_reorder_manual_s4096.svg)

![S=16384 runtime by sparse type](reports/raw_newest_reorder_manual_s16384.svg)

注：`S=16384` 图中的橙色/绿色柱使用公平 A/B 数据，即 `Identity external` / `Reorder external`；Raw 和 Manual 仍使用原四路径矩阵数据。

### 5.2 关键观察

- `causal` 不启用 reorder；Newest causal fastpath 在 `S=1024/4096/16384` 分别为 `1.2810x / 1.2748x / 1.1237x`。
- 端到端表中，1K/4K 的 reorder/external 仍混入 external/FULL_KV 路径差异；16K 图和表已改用 optimized fair A/B。
- 最新 16K optimized fair A/B 显示：`strided_bs` `1.0363x`、`prefix_lm_bs` `1.0313x`、`nested_bs` `1.0290x`、`hybrid_sparse_bs` `1.0195x`，均 allclose 通过。
- `global_local_bs` 的端到端 `2.4x-3.3x` 大幅收益不能归因于 row reorder；最新 16K 公平 A/B 为 `0.9979x`，主要收益来自 external FULL_KV / pure block-sparse 路径差异。
- Manual 在小/中序列部分模式有竞争力，尤其 `causal@1024/4096`；但 16K 下除 causal 外多数 FULL_KV sparse manual 已明显慢于 Newest。

### 5.3 详细数据

| Seq Len | Config | Raw flex | Newest (without reorder) | Newest (with reorder) | Manual | Raw/Newest(no reorder) | Newest(no reorder)/Newest(with reorder) | Manual/Newest(no reorder) | Notes |
|--------:|--------|---------:|-------------------------:|----------------------:|-------:|-----------------------:|----------------------------------------:|--------------------------:|-------|
| 1024 | `band_global_bs` | 0.380 | 0.382 | 0.365 | 0.407 | 0.9948x | 1.0466x | 1.0654x | - |
| 1024 | `block_diagonal_64_bs` | 0.383 | 0.428 | 0.387 | 0.380 | 0.8949x | 1.1059x | 0.8879x | manual 更快 |
| 1024 | `causal` | 0.620 | 0.484 | SKIP | 0.356 | 1.2810x | - | 0.7355x | causal 不启用 reorder; manual 更快 |
| 1024 | `checkerboard_64_bs` | 0.384 | 0.406 | 0.394 | 1.180 | 0.9458x | 1.0305x | 2.9064x | - |
| 1024 | `dilated_window_bs` | 0.423 | 0.377 | 0.368 | 0.373 | 1.1220x | 1.0245x | 0.9894x | manual 基本持平 |
| 1024 | `global_local_bs` | 0.583 | 0.521 | 0.390 | 0.406 | 1.1190x | 1.3359x | 0.7793x | manual 快于 Newest |
| 1024 | `hybrid_sparse_bs` | 0.375 | 0.384 | 0.369 | 0.393 | 0.9766x | 1.0407x | 1.0234x | - |
| 1024 | `multiscale_dilated_bs` | 0.388 | 0.402 | 0.359 | 0.403 | 0.9652x | 1.1198x | 1.0025x | - |
| 1024 | `nested_bs` | 0.399 | 0.367 | 0.381 | 0.436 | 1.0872x | 0.9633x | 1.1880x | reorder 略慢 |
| 1024 | `prefix_lm_bs` | 0.397 | 0.376 | 0.367 | 0.407 | 1.0559x | 1.0245x | 1.0824x | - |
| 1024 | `sliding_window_128_bs` | 0.378 | 0.403 | 0.374 | 0.387 | 0.9380x | 1.0775x | 0.9603x | manual 快于 Newest |
| 1024 | `strided_bs` | 0.385 | 0.393 | 0.365 | 0.396 | 0.9796x | 1.0767x | 1.0076x | - |
| 4096 | `band_global_bs` | 0.589 | 0.596 | 0.589 | 0.953 | 0.9883x | 1.0119x | 1.5990x | - |
| 4096 | `block_diagonal_64_bs` | 0.369 | 0.379 | 0.365 | 0.982 | 0.9736x | 1.0384x | 2.5910x | - |
| 4096 | `causal` | 1.967 | 1.543 | SKIP | 0.676 | 1.2748x | - | 0.4381x | causal 不启用 reorder; manual 更快 |
| 4096 | `checkerboard_64_bs` | 1.066 | 1.068 | 1.074 | 0.971 | 0.9981x | 0.9944x | 0.9092x | reorder 略慢; manual 更快 |
| 4096 | `dilated_window_bs` | 0.468 | 0.460 | 0.441 | 0.967 | 1.0174x | 1.0431x | 2.1022x | - |
| 4096 | `global_local_bs` | 1.332 | 1.363 | 0.562 | 0.953 | 0.9773x | 2.4253x | 0.6992x | external FULL_KV 路径收益显著 |
| 4096 | `hybrid_sparse_bs` | 0.729 | 0.719 | 0.714 | 0.946 | 1.0139x | 1.0070x | 1.3157x | - |
| 4096 | `multiscale_dilated_bs` | 0.549 | 0.545 | 0.524 | 0.960 | 1.0073x | 1.0401x | 1.7615x | - |
| 4096 | `nested_bs` | 0.622 | 0.662 | 0.601 | 0.959 | 0.9396x | 1.1015x | 1.4486x | - |
| 4096 | `prefix_lm_bs` | 1.148 | 1.132 | 1.063 | 0.983 | 1.0141x | 1.0649x | 0.8684x | manual 更快 |
| 4096 | `sliding_window_128_bs` | 0.417 | 0.443 | 0.405 | 0.970 | 0.9413x | 1.0938x | 2.1896x | - |
| 4096 | `strided_bs` | 0.725 | 0.741 | 0.694 | 0.967 | 0.9784x | 1.0677x | 1.3050x | - |
| 16384 | `band_global_bs` | 1.464 | 1.429 | 1.439 | 21.407 | 1.0245x | 0.9931x | 14.9804x | reorder 变慢 |
| 16384 | `block_diagonal_64_bs` | 0.598 | 0.588 | 0.570 | 21.443 | 1.0170x | 1.0316x | 36.4677x | 明确收益 |
| 16384 | `causal` | 14.502 | 12.906 | SKIP | 12.911 | 1.1237x | - | 1.0004x | causal 不启用 reorder |
| 16384 | `checkerboard_64_bs` | 11.579 | 11.689 | 11.793 | 21.072 | 0.9906x | 0.9912x | 1.8027x | reorder 变慢 |
| 16384 | `dilated_window_bs` | 0.946 | 0.925 | 0.911 | 21.633 | 1.0227x | 1.0154x | 23.3870x | 明确收益 |
| 16384 | `global_local_bs` | 4.667 | 1.436 | 1.439 | 20.315 | 3.2500x | 0.9979x | 14.1469x | 基本持平 |
| 16384 | `hybrid_sparse_bs` | 4.806 | 4.866 | 4.773 | 23.350 | 0.9877x | 1.0195x | 4.7986x | 明确收益 |
| 16384 | `multiscale_dilated_bs` | 1.282 | 1.274 | 1.265 | 20.523 | 1.0063x | 1.0071x | 16.1091x | 小幅收益 |
| 16384 | `nested_bs` | 3.707 | 3.721 | 3.616 | 22.004 | 0.9962x | 1.0290x | 5.9135x | 明确收益 |
| 16384 | `prefix_lm_bs` | 11.847 | 12.043 | 11.678 | 21.620 | 0.9837x | 1.0313x | 1.7952x | 明确收益 |
| 16384 | `sliding_window_128_bs` | 0.771 | 0.749 | 0.751 | 23.007 | 1.0294x | 0.9973x | 30.7170x | 基本持平 |
| 16384 | `strided_bs` | 6.189 | 6.247 | 6.028 | 21.112 | 0.9907x | 1.0363x | 3.3795x | 明确收益 |

### 5.4 16K 公平 reorder A/B

这张表只比较 **identity external vs optimized reorder external**，两边都有相同的 external Q/metadata 处理，只改变 row/KV order。optimized reorder 使用当前白名单策略：`hybrid_sparse_bs / nested_bs / strided_bs / prefix_lm_bs` 走 `auction_union_fast + snake_inv`，其它模式 fallback 到 `wave_overlap + snake_inv`。

完整结果见 [fair_reorder_optimized_16k.md](fair_reorder_optimized_16k.md)。

| Config | Mode | KV order | Identity ms | Reorder ms | Speedup | allclose | 结论 |
|--------|------|----------|------------:|-----------:|--------:|----------|------|
| `causal` | `SKIP` | `-` | SKIP | SKIP | - | - | causal 不启用 reorder |
| `block_diagonal_64_bs` | `wave_overlap` | `snake_inv` | 0.588 | 0.570 | 1.0316x | True | 明确收益 |
| `checkerboard_64_bs` | `wave_overlap` | `snake_inv` | 11.689 | 11.793 | 0.9912x | True | reorder 变慢 |
| `sliding_window_128_bs` | `wave_overlap` | `snake_inv` | 0.749 | 0.751 | 0.9973x | True | 基本持平 |
| `strided_bs` | `auction_union_fast` | `snake_inv` | 6.247 | 6.028 | 1.0363x | True | 明确收益 |
| `dilated_window_bs` | `wave_overlap` | `snake_inv` | 0.925 | 0.911 | 1.0154x | True | 明确收益 |
| `nested_bs` | `auction_union_fast` | `snake_inv` | 3.721 | 3.616 | 1.0290x | True | 明确收益 |
| `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 4.866 | 4.773 | 1.0195x | True | 明确收益 |
| `global_local_bs` | `wave_overlap` | `snake_inv` | 1.436 | 1.439 | 0.9979x | True | 基本持平 |
| `multiscale_dilated_bs` | `wave_overlap` | `snake_inv` | 1.274 | 1.265 | 1.0071x | True | 小幅收益 |
| `prefix_lm_bs` | `auction_union_fast` | `snake_inv` | 12.043 | 11.678 | 1.0313x | True | 明确收益 |
| `band_global_bs` | `wave_overlap` | `snake_inv` | 1.429 | 1.439 | 0.9931x | True | reorder 变慢 |

## 6. 32K / 80K 长序列

测试配置：`B=1, H=2, D=128`。Raw / Newest / Reorder 使用 `warmup=3, repeat=5`。

### 6.1 图表

![S=32768 long runtime by sparse type](reports/long_32k_80k_s32768.svg)

![S=81920 long runtime by sparse type](reports/long_32k_80k_s81920.svg)

注：`S=32768/81920` 图中的橙色/绿色柱使用公平 A/B 数据，即 `Identity external` / `Reorder external`；Raw 仍使用原长序列矩阵数据。`causal` 没有纳入 reorder fair A/B，仍保留原矩阵口径。

### 6.2 性能表

这张表和 6.1 图表使用同一口径：Raw 来自原长序列矩阵；`Newest (without reorder)` 是 `identity external`；`Newest (with reorder)` 是 optimized/fallback external reorder。`Speedup = without / with`。

| Seq Len | Config | Raw | Newest (without reorder) | Newest (with reorder) | Speedup | allclose | 结论 |
|--------:|--------|----:|-------------------------:|----------------------:|--------:|----------|------|
| 32768 | `causal` | 50.059 | 47.401 | SKIP | - | - | causal 不启用 reorder; Newest/Raw 1.0561x |
| 32768 | `block_diagonal_64_bs` | CRASH | 0.859 | 0.846 | 1.0154x | True | 明确收益 |
| 32768 | `checkerboard_64_bs` | CRASH | 45.092 | 45.453 | 0.9921x | True | reorder 变慢 |
| 32768 | `sliding_window_128_bs` | CRASH | 1.214 | 1.189 | 1.0210x | True | 明确收益 |
| 32768 | `strided_bs` | CRASH | 23.227 | 22.766 | 1.0202x | True | 明确收益 |
| 32768 | `dilated_window_bs` | CRASH | 1.533 | 1.543 | 0.9935x | True | reorder 变慢 |
| 32768 | `nested_bs` | CRASH | 12.637 | 12.402 | 1.0189x | True | 明确收益 |
| 32768 | `hybrid_sparse_bs` | CRASH | 16.903 | 16.669 | 1.0140x | True | 明确收益 |
| 32768 | `global_local_bs` | CRASH | 2.567 | 2.553 | 1.0055x | True | 小幅收益 |
| 32768 | `multiscale_dilated_bs` | CRASH | 2.227 | 2.211 | 1.0072x | True | 小幅收益 |
| 32768 | `prefix_lm_bs` | CRASH | 45.955 | 45.428 | 1.0116x | True | 明确收益 |
| 32768 | `band_global_bs` | CRASH | 2.567 | 2.595 | 0.9892x | True | reorder 变慢 |
| 81920 | `causal` | ERR(-9) | ERR(-9) | SKIP | - | - | causal 不启用 reorder; 80K 原矩阵未跑通 |
| 81920 | `block_diagonal_64_bs` | ERR(-9) | 1.669 | 1.660 | 1.0054x | True | 小幅收益 |
| 81920 | `checkerboard_64_bs` | ERR(-9) | 277.973 | 279.488 | 0.9946x | True | reorder 变慢 |
| 81920 | `sliding_window_128_bs` | ERR(-9) | 2.547 | 2.531 | 1.0063x | True | 小幅收益 |
| 81920 | `strided_bs` | ERR(-9) | 140.296 | 140.328 | 0.9998x | True | 基本持平 |
| 81920 | `dilated_window_bs` | ERR(-9) | 3.409 | 3.422 | 0.9962x | True | reorder 变慢 |
| 81920 | `nested_bs` | ERR(-9) | 72.557 | 72.218 | 1.0047x | True | 小幅收益 |
| 81920 | `hybrid_sparse_bs` | ERR(-9) | 98.913 | 98.554 | 1.0036x | True | 小幅收益 |
| 81920 | `global_local_bs` | ERR(-9) | 5.995 | 5.979 | 1.0027x | True | 基本持平 |
| 81920 | `multiscale_dilated_bs` | ERR(-9) | 5.134 | 5.117 | 1.0033x | True | 小幅收益 |
| 81920 | `prefix_lm_bs` | ERR(-9) | 279.565 | 277.726 | 1.0066x | True | 小幅收益 |
| 81920 | `band_global_bs` | ERR(-9) | 5.982 | 5.994 | 0.9980x | True | 基本持平 |

### 6.3 公平 correctness A/B

长序列公平 A/B 使用 **identity external output vs optimized reorder output**，两边都走 external Q/metadata 路径，只改变 row/KV order。测试使用 `warmup=3, repeat=5`，容差：`rtol=0.03, atol=0.03`。完整 optimized 明细见 [fair_reorder_optimized_16k_32k_80k.md](fair_reorder_optimized_16k_32k_80k.md)，分段结果见 [fair_reorder_optimized_32k.md](fair_reorder_optimized_32k.md) 和 [fair_reorder_optimized_80k.md](fair_reorder_optimized_80k.md)。

### 6.4 32K/80K repeat=20 稳定性复测

为避免 repeat=5 的偶然波动，新增 [fair_reorder_autotune.md](fair_reorder_autotune.md) 和 [fair_reorder_auction_autotune.md](fair_reorder_auction_autotune.md)，使用 `warmup=5, repeat=20` 复测重点候选。32K 已经出现稳定 `>=1.01x` 的 selector 候选；80K 仍未达到 `>=1.01x`：

| Seq Len | Config | Mode | KV order | Identity | Reorder | Speedup | allclose | 结论 |
|--------:|--------|------|----------|---------:|--------:|--------:|----------|------|
| 32768 | `band_global_bs` | `wave_overlap` | `snake_inv` | 2.544 | 2.554 | 0.9961x | True | 未达标 |
| 32768 | `band_global_bs` | `wave_union_fast` | `snake_inv` | 2.547 | 2.543 | 1.0016x | True | 未达标 |
| 32768 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | 16.852 | 16.665 | 1.0112x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `wave_union_fast` | `snake_inv` | 16.893 | 16.647 | 1.0148x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 16.935 | 16.636 | 1.0180x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 16.908 | 16.650 | 1.0155x | True | 达标 |
| 32768 | `nested_bs` | `wave_overlap` | `snake_inv` | 12.613 | 12.452 | 1.0129x | True | 达标 |
| 32768 | `nested_bs` | `wave_union_fast` | `snake_inv` | 12.622 | 12.479 | 1.0115x | True | 达标 |
| 32768 | `nested_bs` | `auction_union_fast` | `snake_inv` | 12.600 | 12.435 | 1.0133x | True | 达标 |
| 32768 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 12.601 | 12.439 | 1.0130x | True | 达标 |
| 32768 | `strided_bs` | `wave_overlap` | `snake_inv` | 23.186 | 22.810 | 1.0165x | True | 达标 |
| 32768 | `strided_bs` | `wave_union_fast` | `snake_inv` | 23.177 | 22.898 | 1.0122x | True | 达标 |
| 32768 | `strided_bs` | `auction_union_fast` | `snake_inv` | 23.202 | 22.820 | 1.0167x | True | 达标 |
| 32768 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 23.208 | 22.826 | 1.0167x | True | 达标 |
| 81920 | `band_global_bs` | `wave_overlap` | `snake_inv` | 6.022 | 5.978 | 1.0074x | True | 未达标 |
| 81920 | `band_global_bs` | `wave_union_fast` | `snake_inv` | 5.967 | 5.979 | 0.9980x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | 98.863 | 98.431 | 1.0044x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `wave_union_fast` | `snake_inv` | 98.890 | 98.499 | 1.0040x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 98.917 | 98.320 | 1.0061x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 98.961 | 98.188 | 1.0079x | True | 未达标 |
| 81920 | `nested_bs` | `wave_overlap` | `snake_inv` | 72.596 | 72.357 | 1.0033x | True | 未达标 |
| 81920 | `nested_bs` | `wave_union_fast` | `snake_inv` | 72.614 | 72.465 | 1.0021x | True | 未达标 |
| 81920 | `nested_bs` | `auction_union_fast` | `snake_inv` | 72.640 | CRASH | - | - | 未达标 |
| 81920 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 72.588 | 72.162 | 1.0059x | True | 未达标 |
| 81920 | `strided_bs` | `wave_overlap` | `snake_inv` | 140.629 | 140.003 | 1.0045x | True | 未达标 |
| 81920 | `strided_bs` | `wave_union_fast` | `snake_inv` | 140.533 | CRASH | - | - | 未达标 |
| 81920 | `strided_bs` | `auction_union_fast` | `snake_inv` | 140.534 | 140.111 | 1.0030x | True | 未达标 |
| 81920 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 140.701 | 140.554 | 1.0010x | True | 未达标 |

本轮确认 `auction_union_fast` 在 32K `hybrid_sparse_bs / nested_bs / strided_bs` 上是有效迁移，最高 `1.0180x`；`auction_union_exact_path` 能改善部分 80K 稳定性和收益，但最高仍只有 `1.0079x`。因此当前 row/KV reorder 对 32K 已经有可用白名单，对 80K 仍只是小幅正信号，主优化方向应继续转向 FULL_KV / PURE_BLOCK_SPARSE template。

### 6.5 KV orientation 聚焦复测

新增 `union_boundary_dp`，目标是比 `boundary_dp` 的 lo/hi 边界代理更接近专利里的 KV orientation：用真实 wave start/end KV edge set 做 cold-transition DP。完整结果见 [fair_reorder_kv_orientation_focus.md](fair_reorder_kv_orientation_focus.md)。

| Seq Len | Config | Mode | KV order | Identity | Reorder | Speedup | allclose | 结论 |
|--------:|--------|------|----------|---------:|--------:|--------:|----------|------|
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 16.901 | 16.800 | 1.0060x | True | 未达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `boundary_dp` | 16.944 | 16.664 | 1.0168x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `union_boundary_dp` | 16.925 | 16.620 | 1.0184x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 16.915 | 16.682 | 1.0140x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `boundary_dp` | 16.887 | 16.660 | 1.0136x | True | 达标 |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `union_boundary_dp` | 16.885 | 16.652 | 1.0140x | True | 达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 98.797 | CRASH | - | - | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `boundary_dp` | 98.720 | CRASH | - | - | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `union_boundary_dp` | 98.905 | CRASH | - | - | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 98.902 | 98.164 | 1.0075x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `boundary_dp` | 98.920 | 98.286 | 1.0065x | True | 未达标 |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `union_boundary_dp` | 98.847 | 98.167 | 1.0069x | True | 未达标 |

结论：NPU FULL_KV template 对 KV orientation 有响应，32K 上 `union_boundary_dp` 明确优于本轮 `snake_inv`；但 80K 仍没有稳定超过 `1%`，且 `auction_union_fast` 在 80K hybrid 上 crash。因此 `union_boundary_dp` 暂不进入默认 selector，后续应优先做 template 层验证，而不是继续扩大 80K reorder 白名单。

### 6.6 msprof 定位: 32K strided

按照 `.claude/skills/Flex_attn_profiling.md` 的口径，对 `strided_bs@32768` 做了 `identity external` vs `wave_overlap + snake_inv external` 的 msprof 对比。完整记录见 [msprof_strided32k_reorder_analysis.md](msprof_strided32k_reorder_analysis.md)。

| Metric | Identity external | Reorder external | Speedup |
|---|---:|---:|---:|
| `time_runner` avg | 23.347 ms | 22.925 ms | 1.0184x |
| msprof op total | 190.4 ms | 186.9 ms | 1.0187x |
| `triton_tem_fused_0` avg | 22.918 ms | 22.550 ms | 1.0163x |
| helper total | 6.997 ms | 6.456 ms | 1.0838x |
| AI CPU total | 6.363 ms | 5.810 ms | 1.0952x |

结论：32K `strided_bs` 的 reorder 提升主要来自主 fused kernel 变快，而不是 helper 或 launch 减少。`triton_tem_fused_0` 占总 op 时间约 `96%`，cube utilization 已接近满载，因此 80K 若要稳定超过 `1%`，更可能需要 FULL_KV / PURE_BLOCK_SPARSE template 层面的优化，而不是继续只调 row permutation。

### 6.7 msprof 定位: 32K hybrid KV orientation

新增 [msprof_hybrid32k_template_analysis.md](msprof_hybrid32k_template_analysis.md)，对 `hybrid_sparse_bs@32768` 做 `identity external + asc` vs `auction_union_fast + union_boundary_dp` 的 template 层对比：

| Metric | Identity external | Reorder external | Speedup |
|---|---:|---:|---:|
| `time_runner` avg | 17.083 ms | 16.759 ms | 1.0193x |
| msprof op total | 139.9 ms | 137.9 ms | 1.0145x |
| msprof task total | 139.3 ms | 137.2 ms | 1.0153x |
| `triton_tem_fused_0` avg | 16.644 ms | 16.385 ms | 1.0158x |
| helper total | 6.733 ms | 6.792 ms | 0.9913x |

结论：`FULL_KV_IDX` 顺序确实会进入 fused kernel 的 K/V pointer advance，`union_boundary_dp` 的收益主要来自 `triton_tem_fused_0` 变快，而不是 helper 变少。80K identity external 的 msprof 采集触发 `exitCode:11` 且没有 device summary，因此下一步 80K template profile 需要用更小 repeat 或分段采集。

### 6.8 FULL_KV traversal / BLOCK_N probe

新增 [fullkv_traversal_blockn_probe.md](fullkv_traversal_blockn_probe.md)，专门验证 `get_offset_for_next_block()` / FULL_KV traversal 是否是 80K 更直接的优化点。Probe 使用 `hybrid_sparse_bs`、`auction_union_exact_path + union_boundary_dp`、`warmup=3, repeat=5`：

| Seq Len | BLOCK_N | Identity external | Reorder external | Speedup | allclose | 结论 |
|--------:|--------:|------------------:|-----------------:|--------:|----------|------|
| 32768 | 64 | 17.136 | 16.667 | 1.0281x | True | reorder 有收益 |
| 81920 | 64 | 98.951 | CRASH | - | - | 80K 稳定性不足 |
| 32768 | 128 | 10.881 | CRASH | - | - | tile 变快但 reorder 不稳定 |
| 81920 | 128 | 62.938 | 62.981 | 0.9993x | True | identity 明显变快，但 reorder 无收益 |

额外小 shape 校验：`hybrid_sparse_bs@4096` 的 `BLOCK_N=64` vs `BLOCK_N=128` identity external 输出 `allclose=True`，`max_abs=0.001953125`。这说明 `BLOCK_N=128` 有 template 性能信号，但不能直接作为 selector 方案；它更像是在提示默认 `BLOCK_N=64` 下的 FULL_KV traversal 存在可优化空间。

长序列结论：

- Raw 和 Newest without reorder 在 32K/80K 非 causal sparse 上没有稳定完成，因此不能作为公平 speedup 分母。
- 已通过 correctness 的行均 `allclose=True`，最大绝对误差不超过 `0.003906`。
- 本轮 optimized fair repeat=5 下，32K 有多行 `>=1.01x`：`strided_bs 1.0202x`、`nested_bs 1.0189x`、`hybrid_sparse_bs 1.0140x`、`prefix_lm_bs 1.0116x`；80K 补测后没有 `>=1.01x` 行，最高为 `prefix_lm_bs 1.0066x`，重点模式仍只能算小幅正信号或持平。
- 32K repeat=20 下 reorder 本身已经有可用白名单：`hybrid_sparse_bs auction_union_fast 1.0180x`、`strided_bs auction_union_fast 1.0167x`、`nested_bs auction_union_fast 1.0133x`。
- 80K repeat=20 下仍只有小幅正信号，最高为 `hybrid_sparse_bs auction_union_exact_path 1.0079x`；新增 `union_boundary_dp` 聚焦复测最高 `1.0075x`，没有稳定达到 `1%`。
- 因此当前 selector 不默认开启 80K reorder；80K 的下一步主线应是 FlexAttention template 优化，尤其默认 `BLOCK_N=64` 下的 `get_offset_for_next_block()` / FULL_KV traversal，减少非 sparse-block 边界 step 的冗余 metadata load。

## 7. 历史 B/C 消融结论

早期 B/C 消融把 PURE_BLOCK_SPARSE 与 wave_overlap reorder 拆开测：

- Path B = PURE_BLOCK_SPARSE，无 reorder。
- Path C = PURE_BLOCK_SPARSE + wave_overlap reorder。
- B/C 用于估计 row reorder 单独贡献。

历史结果显示，在 `S=32768/49152/65536`、`B=2,H=4` 配置下，B/C 基本落在 `0.995x-1.010x`，多数是噪音级收益或退化。修复 spectral sort / wave partition / intra-wave NNZ sort / inter-wave scheduling 后，32K B/C 仍主要在 ±1% 内。

这说明 GPU 专利里的 wave scheduling 思路迁到 NPU 后，**不能直接期待普适加速**。NPU 当前最重要的收益仍来自 FULL_KV / PURE_BLOCK_SPARSE 模板；reorder 只在少数模式和长序列上出现小幅正信号。

## 8. 当前风险与下一步

风险：

- external reorder 和普通 no-reorder 路径不同，长序列性能表不能直接解释为普通 no-reorder 到 reorder 的 speedup。
- 早期 correctness 复跑中出现过 reorder external crash；当前最终性能表以 `allclose=True` 的稳定完成行为准。
- block-level metadata 会牺牲 token-level mask 的细粒度表达能力；对 block-level pattern 等价，对精确 token-level pattern 可能有最多一个 block 边界误差。

下一步建议：

1. 缓存 `perm + reordered metadata`，减少 host 侧 reorder 开销。
2. `auction_union_fast + exact_path` 离线/bitset 版本已经实现并完成 repeat=20 复测；32K 有收益，80K 仍未达标。下一步优先做 template 层优化，而不是继续默认扩大 reorder 白名单。
3. 转向 FULL_KV / PURE_BLOCK_SPARSE template 优化，优先检查默认 `BLOCK_N=64` 下的 `get_offset_for_next_block()`、FULL_KV metadata traversal 和 80K msprof 采集稳定性；`BLOCK_N=128` 只作为诊断信号，不进入 selector。
4. 保留 selector 白名单策略：只对 correctness 通过且 repeat=20 稳定 `>=1.01x` 的组合启用。
5. 继续保留 causal fastpath，不把 causal 纳入 reorder 优化目标。

## 9. 测试配置

- 设备: NPU Ascend 60GB
- 精度: bfloat16
- 编译: `torch.compile(backend="inductor", dynamic=False)`
- Block size: `SPARSE_Q_BLOCK_SIZE=128`, `SPARSE_KV_BLOCK_SIZE=128`
- 短中序列表: `B=1,H=2,D=128`, flex/reorder `warmup=3, repeat=5`, manual `warmup=1, repeat=3`
- 长序列表: `B=1,H=2,D=128`, raw/newest/reorder `warmup=3, repeat=5`, manual `warmup=1, repeat=3`
- 16K 公平 A/B: identity external vs optimized reorder external, `warmup=3, repeat=5`, `rtol=0.03, atol=0.03`
- 32K/80K 公平 A/B: identity external vs optimized reorder external, `warmup=3, repeat=5`, `rtol=0.03, atol=0.03`
- 32K/80K 稳定性复测: identity external vs reorder external, `warmup=5, repeat=20`, `rtol=0.03, atol=0.03`
- KV orientation 聚焦复测: `hybrid_sparse_bs`, `auction_union_fast/exact_path`, `snake_inv/boundary_dp/union_boundary_dp`, `warmup=5, repeat=20`
