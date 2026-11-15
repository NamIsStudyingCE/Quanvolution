#!/usr/bin/env bash
set -euo pipefail

# On-board hardware benchmark for NVIDIA Jetson Nano (ARM Cortex-A57 @ 1.43 GHz, 2 MB L2 Cache)
# Zero Python / Zero PyTorch / Zero CUDA dependencies required.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/medquanv-cli"
WEIGHTS="${SCRIPT_DIR}/sample_test_data/wide_v2_pneumoniamnist_s42_rust.bin"
VERIFY="${SCRIPT_DIR}/sample_test_data/pneumoniamnist_100_rust_verify.bin"
REPORT_TXT="${SCRIPT_DIR}/jetson_hardware_report.txt"
REPORT_JSON="${SCRIPT_DIR}/jetson_hardware_report.json"

if [ ! -x "${BIN}" ]; then
    chmod +x "${BIN}" || true
fi

{
    echo "========================================================================"
    echo "  JETSON NANO HARDWARE BENCHMARK REPORT"
    echo "  Timestamp: $(date -Iseconds)"
    echo "========================================================================"
    echo ""
    echo "--- 1. CPU & Cache Topology (lscpu) ---"
    lscpu || uname -a
    echo ""
    if command -v nvpmodel >/dev/null 2>&1; then
        echo "--- 2. Current Power Mode (nvpmodel -q) ---"
        nvpmodel -q || true
        echo ""
    fi

    echo "--- 3. End-to-End Inference & Memory Footprint (/usr/bin/time -v) ---"
    if command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -v "${BIN}" --weights "${WEIGHTS}" --verify "${VERIFY}" --benchmark --json-out "${REPORT_JSON}" 2>&1
    else
        "${BIN}" --weights "${WEIGHTS}" --verify "${VERIFY}" --benchmark --json-out "${REPORT_JSON}"
    fi

    if command -v tegrastats >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        echo ""
        echo "--- 4. Power & Clock Telemetry (tegrastats, 10W & 5W modes) ---"
        for mode_id in 0 1; do
            echo "[nvpmodel -m ${mode_id}]"
            sudo nvpmodel -m "${mode_id}" || true
            sudo tegrastats --interval 500 --logfile "${SCRIPT_DIR}/tegrastats_m${mode_id}.log" &
            T_PID=$!
            "${BIN}" --weights "${WEIGHTS}" --verify "${VERIFY}" --benchmark >/dev/null
            sleep 1
            sudo kill "${T_PID}" 2>/dev/null || true
            tail -n 3 "${SCRIPT_DIR}/tegrastats_m${mode_id}.log" || true
        done
    else
        echo ""
        echo "[Note] tegrastats/sudo not available; skipped 5W/10W power-mode sweep."
    fi
} | tee "${REPORT_TXT}"

echo ""
echo "[Done] Saved ${REPORT_TXT} and ${REPORT_JSON}"
