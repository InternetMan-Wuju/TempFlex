#!/usr/bin/env bash
# Systematic correctness + performance benchmark: Raw vs Newest
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH"
DEVICE="npu:0"
WARMUP=3
REPEAT=5
TIMEOUT=120

SCRIPT="python3 flex_attention_run_script.py"

# Patterns that work on both raw and Newest
COMMON_PATTERNS=(causal global_local random_block_sparse)

# FULL_KV patterns (Newest only)
FULLKV_PATTERNS=(block_diagonal_64_bs sliding_window_128_bs nested_bs strided_bs \
  checkerboard_64_bs dilated_window_bs hybrid_sparse_bs prefix_lm_bs)

run_one() {
    local target="$1" seq_len="$2" cfg="$3"
    local result
    result=$(timeout $TIMEOUT $SCRIPT --sparse-config "$cfg" --seq-len "$seq_len" \
        --device "$DEVICE" --target "$target" --warmup $WARMUP --repeat $REPEAT \
        --no-trim-outliers 2>&1)
    if [ $? -ne 0 ]; then
        echo "TIMEOUT/CRASH"
        return
    fi
    local flex_time=$(echo "$result" | grep "Flex Attention avg" | grep -oP '\d+\.\d+')
    local manual_time=$(echo "$result" | grep "Manual Attention avg" | grep -oP '\d+\.\d+')
    local allclose=$(echo "$result" | grep -c "测试通过" || true)
    local fail_rate=$(echo "$result" | grep "fail_ratio" | grep -oP '\d+\.\d+%' || echo "N/A")
    echo "${flex_time:-N/A}|${manual_time:-N/A}|${allclose}|${fail_rate}"
}

echo "============================================"
echo "RAW FLEX Benchmark"
echo "============================================"
bash raw_flex/apply_raw.sh 2>&1 | tail -1

echo ""
echo "--- S=1024 Correctness (flex vs manual) ---"
printf "%-25s %10s %10s %6s %s\n" Pattern Flex_ms Manual_ms Allclose FailRate
for cfg in "${COMMON_PATTERNS[@]}"; do
    result=$(run_one both 1024 "$cfg")
    IFS='|' read -r flex man ok fr <<< "$result"
    printf "%-25s %10s %10s %6s %s\n" "$cfg" "$flex" "$man" "$ok" "$fr"
done

echo ""
echo "--- S=1024 Performance (flex only) ---"
printf "%-25s %10s\n" Pattern Flex_ms
for cfg in "${COMMON_PATTERNS[@]}"; do
    result=$(run_one flex 1024 "$cfg")
    flex=$(echo "$result" | cut -d'|' -f1)
    printf "%-25s %10s\n" "$cfg" "$flex"
done

echo ""
echo "--- S=8192 Performance (flex only) ---"
printf "%-25s %10s\n" Pattern Flex_ms
for cfg in "${COMMON_PATTERNS[@]}"; do
    result=$(run_one flex 8192 "$cfg")
    flex=$(echo "$result" | cut -d'|' -f1)
    printf "%-25s %10s\n" "$cfg" "$flex"
done

echo ""
echo "============================================"
echo "NEWEST Benchmark"
echo "============================================"
bash Newest/apply_newest.sh 2>&1 | tail -1

echo ""
echo "--- S=1024 Correctness (flex vs manual) ---"
printf "%-25s %10s %10s %6s %s\n" Pattern Flex_ms Manual_ms Allclose FailRate
for cfg in "${COMMON_PATTERNS[@]}" "${FULLKV_PATTERNS[@]}"; do
    result=$(run_one both 1024 "$cfg")
    IFS='|' read -r flex man ok fr <<< "$result"
    printf "%-25s %10s %10s %6s %s\n" "$cfg" "$flex" "$man" "$ok" "$fr"
done

echo ""
echo "--- S=8192 Performance (flex only) ---"
printf "%-25s %10s\n" Pattern Flex_ms
for cfg in "${COMMON_PATTERNS[@]}" "${FULLKV_PATTERNS[@]}"; do
    result=$(run_one flex 8192 "$cfg")
    flex=$(echo "$result" | cut -d'|' -f1)
    printf "%-25s %10s\n" "$cfg" "$flex"
done

echo ""
echo "DONE"
