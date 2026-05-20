#!/usr/bin/env bash
set -euo pipefail
# compare_flex.sh — A/B 对比原始 vs 最新 flex_attention
# 用法：bash compare_flex.sh [shape_suite]

SUITE="${1:-small}"   # single / small / smoke

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
RUN_SCRIPT="${ROOT}/flex_attention_run_script.py"

RED='\033[31m'
GREEN='\033[32m'
CYAN='\033[36m'
RESET='\033[0m'

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "[ERROR] 找不到 flex_attention_run_script.py" >&2
    exit 1
fi

# ── 清理缓存，确保每次从头编译 ──
clear_cache() {
    rm -rf /tmp/torchinductor_probe 2>/dev/null || true
    rm -rf "$HOME/.triton/cache" 2>/dev/null || true
}

run_bench() {
    local label="$1"
    echo -e "${CYAN}========================================${RESET}"
    echo -e "${CYAN}  ${label}${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    clear_cache
    python3 "$RUN_SCRIPT" \
        --mode benchmark \
        --target both \
        --shape-suite "$SUITE" \
        --warmup 5 \
        --repeat 20 \
        --device auto \
        2>&1 | tee "/tmp/flex_bench_${label}.log"
    echo ""
}

echo -e "${GREEN}部署原始版...${RESET}"
bash "${SCRIPT_DIR}/apply_raw.sh"

run_bench "raw"

echo -e "${GREEN}部署最新版...${RESET}"
bash "${ROOT}/Newest/apply_newest.sh"

run_bench "newest"

# ── 提取对比 ──
echo -e "${CYAN}========================================${RESET}"
echo -e "${CYAN}  速度对比摘要${RESET}"
echo -e "${CYAN}========================================${RESET}"

extract_ms() {
    local log="$1"
    local target="$2"
    grep -i "${target}" "$log" | grep -oP 'avg:\s*\K[0-9.]+' | head -1
}

RAW_FLEX=$(extract_ms "/tmp/flex_bench_raw.log" "Flex Attention")
NEW_FLEX=$(extract_ms "/tmp/flex_bench_newest.log" "Flex Attention")
RAW_MAN=$(extract_ms "/tmp/flex_bench_raw.log" "Manual Attention")
NEW_MAN=$(extract_ms "/tmp/flex_bench_newest.log" "Manual Attention")

echo ""
printf "  %-25s %s\n" "指标" "raw → newest"
printf "  %-25s %s\n" "-------------------------" "-------------------------"
if [[ -n "$RAW_FLEX" && -n "$NEW_FLEX" ]]; then
    printf "  %-25s ${RED}%.3f ms${RESET} → ${GREEN}%.3f ms${RESET}\n" \
        "Flex Attention" "$RAW_FLEX" "$NEW_FLEX"
fi
if [[ -n "$RAW_MAN" && -n "$NEW_MAN" ]]; then
    printf "  %-25s ${RED}%.3f ms${RESET} → ${RED}%.3f ms${RESET}\n" \
        "Manual Attention" "$RAW_MAN" "$NEW_MAN"
fi
echo ""

# flex 自己对比（Manual 应该不变，因为 manual 不走编译）
if [[ -n "$RAW_FLEX" && -n "$NEW_FLEX" ]]; then
    SPEEDUP=$(python3 -c "print(f'{(float($RAW_FLEX)/float($NEW_FLEX)):.2f}x')")
    echo "  Flex 加速比 (raw/newest): ${SPEEDUP}"
fi

echo ""
echo "完整日志: /tmp/flex_bench_raw.log  /tmp/flex_bench_newest.log"
