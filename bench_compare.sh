#!/usr/bin/env bash
# 3-way comparison: Raw vs Newest vs Manual (correctness + performance)
# S=1024, warmup=10, repeat=10
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH"
DEVICE="npu:0"
WARMUP=10
REPEAT=10
TIMEOUT=120
SCRIPT="python3 flex_attention_run_script.py"

# Patterns tested
RAW_PATTERNS=(causal random_block_sparse)
NEW_PATTERNS=(causal random_block_sparse block_diagonal_64_bs sliding_window_128_bs \
  nested_bs strided_bs dilated_window_bs hybrid_sparse_bs checkerboard_64_bs prefix_lm_bs)

parse_num() { sed 's/\x1b\[[0-9;]*m//g' | grep "$1" | grep -oE '[0-9]+\.[0-9]+' | head -1; }

run_test() {
    local target="$1" seq_len="$2" cfg="$3"
    timeout $TIMEOUT $SCRIPT --sparse-config "$cfg" --seq-len "$seq_len" \
        --device "$DEVICE" --target "$target" --warmup $WARMUP --repeat $REPEAT \
        --no-trim-outliers 2>&1
}

echo "============================================"
echo "3-Way Comparison: Raw vs Newest vs Manual"
echo "S=$seq_len  warmup=$WARMUP  repeat=$REPEAT"
echo "============================================"

# ── RAW ──
echo ""
echo "### RAW FLEX ###"
bash raw_flex/apply_raw.sh 2>&1 | tail -1
for cfg in "${RAW_PATTERNS[@]}"; do
    echo "--- $cfg ---"
    out=$(run_test both 1024 "$cfg")
    flex=$(echo "$out" | parse_num "Flex Attention avg")
    man=$(echo "$out" | parse_num "Manual Attention avg")
    ok=$(echo "$out" | grep -c "测试通过" || echo "0")
    echo "  flex=${flex:-FAIL} ms  manual=${man:-FAIL} ms  allclose=$ok"
done

# ── NEWEST ──
echo ""
echo "### NEWEST FLEX ###"
bash Newest/apply_newest.sh 2>&1 | tail -1
for cfg in "${NEW_PATTERNS[@]}"; do
    echo "--- $cfg ---"
    out=$(run_test both 1024 "$cfg")
    flex=$(echo "$out" | parse_num "Flex Attention avg")
    man=$(echo "$out" | parse_num "Manual Attention avg")
    ok=$(echo "$out" | grep -c "测试通过" || echo "0")
    echo "  flex=${flex:-FAIL} ms  manual=${man:-FAIL} ms  allclose=$ok"
done

echo ""
echo "DONE"
