#!/usr/bin/env bash
set -euo pipefail

# ===============================
# 添加 Python3.11 动态库路径
# ===============================
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

copy_one() {
    local src="$1"
    local dst="$2"

    if [[ ! -f "$src" ]]; then
        echo "[ERROR] missing source: ${src}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"   # 只替换，不做任何 backup
    echo "[INSTALL] ${src} -> ${dst}"
}

echo "[INFO] applying flex attention operator (no backup) ..."

copy_one "${SCRIPT_DIR}/site-packages/torch_npu/_inductor/kernel/flex_attention.py" \
         "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py"

echo "[OK] flex attention operator replaced"

copy_one "${SCRIPT_DIR}/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py" \
         "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py"

echo "[OK] flex attention reorder deployed"