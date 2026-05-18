# ===============================
# 添加 Python3.11 动态库路径
# ===============================
PYTHON_LIB_DIR="/usr/local/python3.11.14/lib"
if [ -f "$PYTHON_LIB_DIR/libpython3.11.so.1.0" ]; then
    export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"
    echo "[OK] 已将 python3.11 动态库路径加入 LD_LIBRARY_PATH"
else
    echo "[ERROR] 未找到 libpython3.11.so.1.0，请检查 Python 安装！"
    #exit 1
fi

ln -sf /usr/local/python3.11.14/bin/python3.11 /usr/bin/python3