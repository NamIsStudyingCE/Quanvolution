#!/usr/bin/env bash
set -euo pipefail

# Cross-compile standalone ARM64 binary & static library for NVIDIA Jetson Nano
# Target: Quad-core ARM Cortex-A57 @ 1.43 GHz (ARMv8 NEON, 2 MB L2 Cache)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUST_DIR="${REPO_ROOT}/rust_qlut"
DIST_DIR="${REPO_ROOT}/dist/jetson_nano"

export RUSTFLAGS="-C target-cpu=cortex-a57 -C opt-level=3 -C lto=fat -C codegen-units=1"

mkdir -p "${DIST_DIR}/sample_test_data"

cd "${RUST_DIR}"

if command -v cargo-zigbuild >/dev/null 2>&1; then
    echo "[build_jetson_release] Using cargo-zigbuild (aarch64-unknown-linux-gnu.2.27 for JetPack 4.x glibc compatibility)..."
    cargo zigbuild --release --no-default-features --target aarch64-unknown-linux-gnu.2.27 --lib --bin medquanv-cli
    TARGET_OUT="${RUST_DIR}/target/aarch64-unknown-linux-gnu/release"
else
    echo "[build_jetson_release] Using cargo --target aarch64-unknown-linux-gnu..."
    cargo build --release --no-default-features --target aarch64-unknown-linux-gnu --lib --bin medquanv-cli
    TARGET_OUT="${RUST_DIR}/target/aarch64-unknown-linux-gnu/release"
fi

cp -f "${TARGET_OUT}/medquanv-cli" "${DIST_DIR}/medquanv-cli"
cp -f "${TARGET_OUT}/librust_qlut.a" "${DIST_DIR}/librust_qlut.a"
chmod +x "${DIST_DIR}/medquanv-cli"

echo "[build_jetson_release] Exported:"
ls -lh "${DIST_DIR}/medquanv-cli" "${DIST_DIR}/librust_qlut.a" "${DIST_DIR}/rust_qlut.h"
