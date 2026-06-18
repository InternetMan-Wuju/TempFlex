# Patent Reorder Migration Gaps

> Date: 2026-06-18  
> Scope: compare GPU patent/source reorder implementation with the current NPU FlexAttention external reorder path.

## Current Migration Status

The current NPU path has already migrated the safe row/KV-order pieces:

| Patent/source idea | Current NPU status | Notes |
|---|---|---|
| `wave_overlap` spectral row order | Implemented in `torch_npu._inductor.kernel.flex_attention_reorder` | Used by default external reorder. |
| Fiedler / local Fiedler variants | Implemented in Newest reorder module | Available in registry, but not selected for 80K by default. |
| Intra-wave NNZ descending | Implemented | Preserves wave load balance. |
| KV order `asc/desc/snake/snake_inv` | Mostly implemented | NPU runner extends this with `boundary_dp` and `edge_dp`. |
| Host-side `banded_union_wave` | Implemented in `flex_attention_run_script.py` | Correct but too slow for hot path. |
| `wave_union_fast` approximation | Implemented in `flex_attention_run_script.py` | Low-overhead interval-center approximation; 80K repeat=20 did not reach 1%. |
| `auction_union_fast` bitset row packing | Implemented in `flex_attention_run_script.py` | 32K repeat=20 reaches up to 1.0180x; 80K still below 1%. |
| `auction_union_exact_path` macro-wave sequencing | Implemented in `flex_attention_run_script.py` | Improves some 80K cases, best 1.0079x on `hybrid_sparse_bs@81920`; not enough for selector. |
| `union_boundary_dp` KV orientation | Implemented in `flex_attention_run_script.py` | Uses real wave start/end KV edge sets; 32K hybrid reaches 1.0184x, 80K still below 1%. |
| Backend-neutral `ReorderPlan` | Implemented in `flex_attention_run_script.py` | Describes `q_perm`, `inv_perm`, `wave_id`, and `kv_orientation` for future direct mask/score integration. |

The current missing pieces are not basic row reorder anymore. Bitset union scoring, macro-wave sequencing, and a more patent-like KV orientation DP have now been migrated as offline/external modes. The remaining high-value gaps are adaptive telemetry/selector, direct mask/score integration, and template-level optimization for the NPU fused kernel.

## High-Value Gaps To Migrate

### 1. Bitset Auction Union Reorder

Source: `greedy_reorder_cuda.py` implements `auction_union_reorder_cuda`, which packs the block mask into bitsets, then assigns rows to fixed-capacity waves by exact marginal KV-union cost.

Why it matters:

- It is closer to the real objective than `wave_union_fast`: minimize new KV columns added to each wave.
- It avoids the slow dense bool `m_h[pool] & ~wave_union` loop in current `banded_union_wave`.
- It keeps the hard skeleton that NPU currently needs: fixed-size waves, NNZ-descending rows inside each wave, macro-local grouping.

NPU adaptation status:

- Implemented as CPU/offline bitset first, not as NPU kernel.
- Uses Python `int` bitsets, which is suitable for current 80K block metadata (`NB=640`).
- CLI mode added: `--block-reorder-mode auction_union_fast`.
- Output remains a normal permutation consumed by the existing external Q/metadata reorder path.

Measured result:

- 32K repeat=20: `hybrid_sparse_bs 1.0180x`, `nested_bs 1.0133x`, `strided_bs 1.0167x`.
- 80K repeat=20: best `hybrid_sparse_bs 1.0061x`; `nested_bs` fast variant crashed.
- Good candidate for 32K selector and offline/cache generation, but not enough to enable 80K by default.

### 2. Macro-Wave Sequencing By Cold-Column Transition Cost

Source has two levels:

- Greedy wave sequencing inside `auction_union_reorder_cuda`: choose wave order by union overlap.
- Exact path ordering for <=8 waves using transition cost `|U_j \ U_i|`.

Why it matters:

- Current NPU `wave_overlap` mostly reorders rows and sorts dense waves first.
- It does not explicitly minimize cold KV columns between adjacent waves.
- 80K repeat=20 showed row permutation alone tops out around `1.004x`; wave sequencing may be the remaining small but real reorder-side lever.

NPU adaptation status:

- Implemented as `--block-reorder-mode auction_union_exact_path`.
- After auction row packing, each macro computes wave union bitsets and runs exact DP path ordering.
- Kept as a standalone mode instead of adding a new combinatorial CLI flag.

Measured result:

- Correctness passed in tested rows, `max_abs <= 0.00390625`.
- 32K repeat=20 remains above 1% on `hybrid_sparse_bs/nested_bs/strided_bs`.
- 80K best is `hybrid_sparse_bs 1.0079x`, still below the stable 1% target.

### 3. Adaptive Elastic Selector

Source `adaptive_elastic_wave_reorder_cuda` computes cheap mask features:

- row NNZ CV: shear pressure.
- column NNZ CV and adjacent Jaccard: overlap pressure.
- candidate window, rank/shear/borrow weights selected from those features.

Why it matters:

- Current selector is mostly empirical whitelist by config and seq length.
- A feature selector can avoid applying reorder to uniform or regression-prone masks, and can choose between `wave_overlap`, `wave_union_fast`, and future `auction_union_fast`.

NPU adaptation:

- Start with no new reorder mode. Add metrics logging: density, row_cv, col_cv, adjacent_jaccard, wave_union_cost, transition_cost.
- Use features to explain selector decisions, not to auto-enable 80K immediately.
- Promote to auto selector only after repeat=20 confirms stable wins.

Expected risk:

- Low if used as telemetry first.
- Medium if used as automatic selector before enough measurements.

### 4. Direct Mask/Score ReorderPlan Integration

Future application path is expected to pass mask/score directly, not always through FULL_KV metadata. The reorder plan therefore must not be tied to metadata materialization.

NPU adaptation status:

- Added a lightweight `ReorderPlan` with `q_perm`, `inv_perm`, `wave_id`, and `kv_orientation`.
- Current FULL_KV external path consumes the plan by gathering Q and rebuilding reordered metadata.
- Future direct mask/score path should consume the same logical plan as block traversal/order hints, while keeping `mask_mod` and `score_mod` token indices logical.

Important constraint:

- Do not make direct mask/score depend on dynamic `inv_perm[token_q]` lookups inside `mask_mod`; that is likely to reintroduce NPU lowering instability.
- Preferred path is to let the template scheduler consume `ReorderPlan` and keep score/mask semantics unchanged.

### 5. Banded Window-Greedy

Source `global_banded_window_greedy_reorder_cuda` uses NNZ banding plus W-lookback overlap greedy.

Why it matters:

- It is the source path described as “best L2 hit rate among tested algorithms”.
- It models a window of recently active rows instead of only the immediately previous row.

NPU adaptation:

- Do not materialize full `S = mask @ mask.T` for 80K.
- If migrated, implement a small-window/offline variant over NNZ bands using bitsets.
- Treat it as an oracle/autotune candidate, not default runtime mode.

Expected risk:

- Higher compute cost than auction/path sequencing.
- More likely useful for offline cache generation than online benchmarking.

## Lower-Priority Or Risky Pieces

### Full Global Greedy / TSP Similarity Matrix

Source `global_greedy_reorder_cuda` precomputes `S = mask @ mask.T`.

Why not migrate now:

- Memory and time grow with `MB^2`.
- For 80K with block size 128, `MB=640`; this is still possible for one case but not attractive as a general NPU runtime path.
- It optimizes an ordering proxy that may not map cleanly to the current NPU fused kernel bottleneck.

Use only as:

- Offline oracle for a few patterns.
- Diagnostic reference to see if better row order exists at all.

### Partition Refinement / Segment Partition

Source has local swap refinement and segment-aware partitioning oracles.

Why not migrate as default:

- The source itself marks these as oracle/research paths.
- They use Python loops and local swaps.
- Correctness is easy, but runtime overhead is not acceptable without cache.

Use only as:

- Offline upper-bound experiment.
- Generate a target cost to compare `auction_union_fast` against.

## Updated Recommendation

1. Add reorder telemetry metrics:
   `density`, `row_cv`, `col_cv`, `adjacent_jaccard`, `mean_wave_union`, `mean_transition_cold_cols`, and `nnz_shear`.

2. Keep `auction_union_fast` in the 32K selector whitelist only where repeat=20 passed:
   `hybrid_sparse_bs`, `nested_bs`, and `strided_bs`.

3. Do not enable 80K reorder by default:
   the best tested auction/exact result is still `1.0079x`; the best `union_boundary_dp` focused result is `1.0075x`, and some modes crash.

4. Move the main 80K effort to FULL_KV / PURE_BLOCK_SPARSE template internals:
   reorder is now showing only sub-1% order sensitivity at 80K.

5. Use `auction_union_exact_path` and `union_boundary_dp` as offline diagnostic candidates:
   useful for proving whether better row/wave order exists, but not a default runtime path yet.

## Expected Best Bet

The best reorder candidate has now been tested:

```text
auction_union_fast + exact_path sequencing
```

This is the closest NPU-friendly migration of the patent’s later production path. It targets exactly what current `wave_union_fast` only approximates: fixed-capacity wave construction that minimizes KV union, followed by macro-wave sequencing that minimizes cold-column transitions.

Current result: it improves 32K, especially with `union_boundary_dp` on `hybrid_sparse_bs@32768`, but still cannot make 80K stable `>=1.01x`. The remaining reorder-side opportunity is probably small for the current external path, so 80K optimization should move to FULL_KV / PURE_BLOCK_SPARSE template internals and future direct mask/score scheduling via `ReorderPlan`.
