#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/$(date +%Y%m%d_%H%M%S)"

copy_one() {
    local src="$1"
    local dst="$2"
    local rel="${dst#/}"
    local backup="${BACKUP_DIR}/${rel}"

    if [[ ! -f "${src}" ]]; then
        echo "[ERROR] missing source: ${src}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${dst}")"
    if [[ -e "${dst}" ]]; then
        mkdir -p "$(dirname "${backup}")"
        cp -a "${dst}" "${backup}"
        echo "[BACKUP] ${dst} -> ${backup}"
    else
        echo "[BACKUP] ${dst} did not exist"
    fi

    cp -a "${src}" "${dst}"
    echo "[INSTALL] ${src} -> ${dst}"
}

echo "[INFO] applying Newest patch bundle from: ${SCRIPT_DIR}"
echo "[INFO] repo root: ${REPO_ROOT}"
echo "[INFO] backup dir: ${BACKUP_DIR}"

copy_one "${SCRIPT_DIR}/workspace/flex_attention2.py" \
    "${REPO_ROOT}/flex_attention2.py"

copy_one "${SCRIPT_DIR}/site-packages/torch_npu/_inductor/kernel/flex_attention.py" \
    "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py"

copy_one "${SCRIPT_DIR}/site-packages/torch_npu/_inductor/npu_triton_heuristics.py" \
    "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/npu_triton_heuristics.py"

copy_one "${SCRIPT_DIR}/site-packages/torch/_inductor/async_compile.py" \
    "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/_inductor/async_compile.py"

echo "[OK] Newest patch bundle applied"
