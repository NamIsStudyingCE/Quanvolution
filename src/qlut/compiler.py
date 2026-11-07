"""
Q-LUT Compiler  (Step 1 của MVE)
================================
Biên dịch một mạch Quanvolution 4-qubit CỐ ĐỊNH (hoặc đã huấn luyện xong rồi "đóng băng")
thành bảng tra cứu:

    table[p0, p1, p2, p3, c] = <Z_c>( mạch(x0..x3) ),   x_i = p_i / (L-1),  L = 2**n_bits

- Thứ tự pixel khớp với `apply_quanv_to_image` / `F.unfold` (row-major trong patch 2x2):
  (p0, p1) = hàng trên, (p2, p3) = hàng dưới.
- Chỉ số tuyến tính: index = (p0 << 3n) | (p1 << 2n) | (p2 << n) | p3  (n = n_bits).
- Mạch được xây lại trên `default.qubit` với broadcasting (cả lô điểm lưới trong 1 lần gọi) dùng
  ĐÚNG trọng số của `src/models/circuits.py`; sau đó đối chiếu với QNode gốc trong CIRCUIT_DICT
  ở một số điểm ngẫu nhiên (`verify_against_reference`) để chứng minh không sai lệch.

Cách chạy:
    python -m src.qlut.compiler --circuit strongly_L2 --bits 4 5
"""
import argparse
import os
import time

import numpy as np
import pennylane as qml
import torch

N_QUBITS = 4
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")


# ----------------------------------------------------------------------------
# Mạch tham chiếu có hỗ trợ broadcasting (dùng chung trọng số với circuits.py)
# ----------------------------------------------------------------------------
def build_batched_qnode(circuit_name: str):
    """QNode (default.qubit) nhận phi có shape (N, 4) và trả về (4, N) kỳ vọng <Z>."""
    from src.models import circuits as C  # import muộn: tránh khởi tạo device khi không cần

    ansatz_map = {
        "random_L1": lambda w: qml.RandomLayers(weights=[C.random_w_1l], wires=range(N_QUBITS)),
        "random_L2": lambda w: qml.RandomLayers(weights=C.random_w_2l, wires=range(N_QUBITS)),
        "strongly_L1": lambda w: qml.StronglyEntanglingLayers(weights=C.strongly_w_1l, wires=range(N_QUBITS)),
        "strongly_L2": lambda w: qml.StronglyEntanglingLayers(weights=C.strongly_w_2l, wires=range(N_QUBITS)),
        "basic_L1": lambda w: qml.BasicEntanglerLayers(weights=C.basic_w_1l, wires=range(N_QUBITS)),
        "basic_L2": lambda w: qml.BasicEntanglerLayers(weights=C.basic_w_2l, wires=range(N_QUBITS)),
    }
    if circuit_name not in ansatz_map:
        raise ValueError(f"circuit '{circuit_name}' không có. Chọn: {list(ansatz_map)}")
    apply_ansatz = ansatz_map[circuit_name]
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def qnode(phi):
        for j in range(N_QUBITS):
            qml.RY(np.pi * phi[:, j], wires=j)
        apply_ansatz(None)
        return [qml.expval(qml.PauliZ(j)) for j in range(N_QUBITS)]

    return qnode


def build_trainable_batched_fn(weights: np.ndarray, ansatz: str = "basic"):
    """Mạch huấn luyện được (trainable_quanv.py) với trọng số θ đã đóng băng -> hàm (N,4)->(N,4)."""
    dev = qml.device("default.qubit", wires=N_QUBITS)
    w = np.asarray(weights)

    @qml.qnode(dev)
    def qnode(phi):
        qml.AngleEmbedding(phi * np.pi, wires=range(N_QUBITS), rotation="Y")
        if ansatz == "strongly":
            qml.StronglyEntanglingLayers(weights=w, wires=range(N_QUBITS))
        else:
            qml.BasicEntanglerLayers(weights=w, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(j)) for j in range(N_QUBITS)]

    return lambda phi: np.stack(qnode(phi), axis=-1)


# ----------------------------------------------------------------------------
# Biên dịch
# ----------------------------------------------------------------------------
def compile_qlut_from_fn(batched_fn, n_bits: int = 4, chunk: int = 1 << 16) -> torch.Tensor:
    """batched_fn: ndarray (N,4) trong [0,1] -> ndarray (N,4). Trả về tensor float32 (L,L,L,L,4)."""
    L = 1 << n_bits
    total = L ** 4
    out = np.empty((total, N_QUBITS), dtype=np.float32)
    levels = np.arange(L, dtype=np.float64) / (L - 1)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        p0 = idx >> (3 * n_bits)
        p1 = (idx >> (2 * n_bits)) & (L - 1)
        p2 = (idx >> n_bits) & (L - 1)
        p3 = idx & (L - 1)
        phi = np.stack([levels[p0], levels[p1], levels[p2], levels[p3]], axis=1)
        out[start:start + len(idx)] = batched_fn(phi)
    return torch.from_numpy(out).view(L, L, L, L, N_QUBITS).contiguous()


def compile_qlut(n_bits: int = 4, circuit: str = "strongly_L2", save: bool = True):
    """Biên dịch mạch tĩnh `circuit` thành Q-LUT. Trả về (table, info dict)."""
    qnode = build_batched_qnode(circuit)
    fn = lambda phi: np.stack(qnode(phi), axis=-1)
    t0 = time.perf_counter()
    table = compile_qlut_from_fn(fn, n_bits)
    compile_s = time.perf_counter() - t0

    info = {
        "circuit": circuit,
        "n_bits": n_bits,
        "levels": 1 << n_bits,
        "compile_seconds": compile_s,
        "table_bytes": table.numel() * table.element_size(),
        "disk_bytes": None,
        "path": None,
    }
    if save:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        path = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_{n_bits}bit.pt")
        torch.save({"table": table, "n_bits": n_bits, "circuit": circuit,
                    "compile_seconds": compile_s}, path)
        info["path"] = path
        info["disk_bytes"] = os.path.getsize(path)
    return table, info


def load_qlut(n_bits: int = 4, circuit: str = "strongly_L2") -> torch.Tensor:
    path = os.path.join(WEIGHTS_DIR, f"qlut_{circuit}_{n_bits}bit.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} chưa có. Chạy: python -m src.qlut.compiler --circuit {circuit} --bits {n_bits}")
    return torch.load(path, weights_only=False)["table"]


# ----------------------------------------------------------------------------
# Kiểm tra tính đúng đắn: bảng biên dịch ≡ QNode gốc trong repo
# ----------------------------------------------------------------------------
def verify_against_reference(table: torch.Tensor, n_bits: int, circuit: str, n_points: int = 50, seed: int = 0):
    """Lấy n_points điểm LƯỚI ngẫu nhiên, chạy QNode GỐC của repo (từng điểm), so với bảng. Trả về max |diff|."""
    from src.models.circuits import CIRCUIT_DICT

    rng = np.random.default_rng(seed)
    L = 1 << n_bits
    p = rng.integers(0, L, size=(n_points, 4))
    worst = 0.0
    for row in p:
        phi = row / (L - 1)
        ref = np.array([float(v) for v in CIRCUIT_DICT[circuit](list(phi))])
        got = table[tuple(row)].numpy()
        worst = max(worst, float(np.abs(ref - got).max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="strongly_L2")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    for b in args.bits:
        table, info = compile_qlut(b, args.circuit, save=True)
        print(f"[compile] circuit={info['circuit']} bits={b} levels={info['levels']} "
              f"entries={info['levels'] ** 4:,}")
        print(f"          compile_time = {info['compile_seconds']:.2f} s")
        print(f"          table (RAM)  = {info['table_bytes'] / 2**20:.3f} MiB  shape={tuple(table.shape)}")
        print(f"          file on disk = {info['disk_bytes'] / 2**20:.3f} MiB  -> {info['path']}")
        if not args.no_verify:
            d = verify_against_reference(table, b, args.circuit)
            print(f"          verify vs repo QNode (50 grid pts): max|diff| = {d:.3e}")


if __name__ == "__main__":
    main()
