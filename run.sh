#!/usr/bin/env bash
# Build, generate data, run the kernel on the NPU and verify the result.
#
#   ./run.sh            # build + run every case in scripts/cases.py
#   ./run.sh 1          # build + run only case 1
#   JOBS=8 ./run.sh     # control build parallelism
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="matmul_custom"
CASE="${1:-all}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set. Please run:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "or set ASCEND_HOME_PATH to your CANN toolkit path."
    exit 1
fi

if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi

echo "=== [1/3] Configure + build ==="
mkdir -p build
cmake -S . -B build
cmake --build build -j "${JOBS}"

echo "=== [2/3] Generate test data ==="
(cd build && python3 ../scripts/gen_data.py)

run_case() {
    local cid="$1"
    echo "=== [3/3] Run case ${cid} ==="
    cd "${SCRIPT_DIR}/build"
    cp -f "input/case${cid}/a.bin" input/a.bin
    cp -f "input/case${cid}/b.bin" input/b.bin
    cp -f "input/case${cid}/dims.txt" input/dims.txt
    cp -f "output/golden_case${cid}/golden_c.bin" output/golden_c.bin

    if timeout 300 "./${OP_NAME}"; then
        if python3 ../scripts/verify_result.py "${cid}"; then
            echo "--- case ${cid}: PASSED ---"
            return 0
        fi
        echo "--- case ${cid}: FAILED (result mismatch) ---"
        return 1
    fi
    echo "--- case ${cid}: FAILED (kernel exited non-zero or timed out) ---"
    return 1
}

if [ "${CASE}" = "all" ]; then
    status=0
    for d in "${SCRIPT_DIR}"/build/input/case*; do
        [ -d "${d}" ] || continue
        cid="${d##*case}"
        run_case "${cid}" || status=1
    done
    if [ "${status}" -ne 0 ]; then
        echo "=== FAILED ==="
        exit 1
    fi
    echo "=== ALL CASES PASSED ==="
else
    run_case "${CASE}"
    echo "=== PASSED ==="
fi
