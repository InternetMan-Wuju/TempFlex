#!/usr/bin/env bash
set -euo pipefail
# apply_raw.sh — 部署原始（未修改）flex_attention.py
# 用法：bash apply_raw.sh

PYTHON_LIB_DIR="/usr/local/python3.11.14/lib"
if [ -f "$PYTHON_LIB_DIR/libpython3.11.so.1.0" ]; then
    export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"
    echo "[OK] 已将 python3.11 动态库路径加入 LD_LIBRARY_PATH"
else
    echo "[ERROR] 未找到 libpython3.11.so.1.0，请检查 Python 安装！" >&2
    exit 1
fi

ln -sf /usr/local/python3.11.14/bin/python3.11 /usr/bin/python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${SCRIPT_DIR}/site-packages/torch_npu/_inductor/kernel/flex_attention.py"
DST="/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py"

if [[ ! -f "$SRC" ]]; then
    echo "[ERROR] 原始文件不存在: ${SRC}" >&2
    echo "请先把原始 flex_attention.py 放到 raw_flex/site-packages/torch_npu/_inductor/kernel/" >&2
    exit 1
fi

mkdir -p "$(dirname "${DST}")"
cp -a "${SRC}" "${DST}"
echo "[INSTALL] 原始版: ${SRC} -> ${DST}"
echo "[OK] 已部署原始（未修改）版 flex_attention"
