#!/usr/bin/env bash
# 3-way comparison: Raw Flex vs Newest Flex vs Manual
# All patterns, S=1024, warmup=10, repeat=10
set -euo pipefail
export LD_LIBRARY_PATH="/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH"

DEV=npU:0 S=1024 W=10 R=10 T=120
SCRIPT="python3 flex_attention_run_script.py"
ALL=(causal random_block_sparse block_diagonal_64_bs sliding_window_128_bs nested_bs strided_bs dilated_window_bs hybrid_sparse_bs checkerboard_64_bs prefix_lm_bs)

parse() { sed 's/\x1b\[[0-9;]*m//g' | grep "$1" | grep -oE '[0-9]+\.[0-9]+' | head -1; }

echo "============================================"
echo "3-Way: Raw vs Newest vs Manual  (S=$S W=$W R=$R)"
echo "============================================"

# ─── RAW ───
bash raw_flex/apply_raw.sh 2>&1 | tail -1
echo ""
printf "%-28s %10s %10s %10s\n" "Pattern" "Raw_ms" "Manual_ms" "Allclose"
echo "----------------------------------------------"
for cfg in "${ALL[@]}"; do
    out=$(timeout $T $SCRIPT --sparse-config "$cfg" --seq-len $S --device $DEV --target both --warmup $W --repeat $R --no-trim-outliers 2>&1)
    f=$(echo "$out" | parse "Flex Attention avg")
    m=$(echo "$out" | parse "Manual Attention avg")
    ok=$(echo "$out" | grep -c "测试通过" || echo "0")
    printf "%-28s %10s %10s %10s\n" "$cfg" "${f:-FAIL}" "${m:-FAIL}" "$ok"
done

# ─── NEWEST ───
bash Newest/apply_newest.sh 2>&1 | tail -1
echo ""
printf "%-28s %10s %10s %10s %10s\n" "Pattern" "Newest_ms" "Raw_ms" "Manual_ms" "Allclose"
echo "---------------------------------------------------------------"
for cfg in "${ALL[@]}"; do
    out=$(timeout $T $SCRIPT --sparse-config "$cfg" --seq-len $S --device $DEV --target both --warmup $W --repeat $R --no-trim-outliers 2>&1)
    f=$(echo "$out" | parse "Flex Attention avg")
    m=$(echo "$out" | parse "Manual Attention avg")
    ok=$(echo "$out" | grep -c "测试通过" || echo "0")
    # Get raw result from previous run
    raw_val=$(grep "^$cfg " /tmp/raw_results.txt 2>/dev/null | awk '{print $2}' || echo "N/A")
    printf "%-28s %10s %10s %10s %10s\n" "$cfg" "${f:-FAIL}" "$raw_val" "${m:-FAIL}" "$ok"
done
echo ""
echo "DONE"