"""
BƯỚC 2: Tiền trích xuất đặc trưng Rust Q-LUT đa tỷ lệ (d=1,2,3,axial -> 16 kênh INT8) cho 3 tập MedMNIST.
Cache: data/processed/{breast,pneumonia}_qlut.npz  (OCT ~340MB nên chỉ đo thời gian, trích xuất tức thời khi cần).
Kèm kiểm tra tính đúng đắn: so khớp cache với trích xuất lại, và ghi thời gian vào reports/pillar2_expansion_extraction.json.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import pillar2_state as st
from models.gr_quanv import extract_multiscale_qlut

SETS = {"breast": "data/breastmnist.npz", "pneumonia": "data/pneumoniamnist.npz", "oct": "data/octmnist.npz"}
CACHE = {"breast": "data/processed/breast_qlut.npz", "pneumonia": "data/processed/pneumonia_qlut.npz"}


def main():
    if st.is_done("step2_feature_extraction") and all(os.path.exists(p) for p in CACHE.values()):
        print("Step 2 already completed -> skip"); return
    report = {}
    for name, path in SETS.items():
        d = np.load(path)
        out, rep = {}, {}
        for split, key in (("train", "train_images"), ("val", "val_images"), ("test", "test_images")):
            u8 = np.ascontiguousarray(d[key], dtype=np.uint8)
            extract_multiscale_qlut(u8[:64], include_axial=True)  # warm-up
            t0 = time.perf_counter()
            q, _ = extract_multiscale_qlut(u8, include_axial=True)
            ms = (time.perf_counter() - t0) * 1000
            q_i8 = np.rint(q.numpy() * 127.0).astype(np.int8)
            rep[split] = dict(n=int(len(u8)), ms=ms, us_per_img=ms * 1000 / len(u8), shape=list(q_i8.shape))
            out[f"q_{split}"] = q_i8
        if name in CACHE:
            np.savez_compressed(CACHE[name], **out)
            chk, _ = extract_multiscale_qlut(np.ascontiguousarray(d["test_images"], dtype=np.uint8), include_axial=True)
            ok = np.array_equal(np.load(CACHE[name])["q_test"], np.rint(chk.numpy() * 127.0).astype(np.int8))
            rep["cache_roundtrip_ok"] = bool(ok)
            assert ok, f"cache mismatch {name}"
        report[name] = rep
        print(name, {k: (round(v["ms"], 2) if isinstance(v, dict) else v) for k, v in rep.items()}, flush=True)
    st.atomic_dump_json(report, "reports/pillar2_expansion_extraction.json")
    st.mark("step2_feature_extraction", True, cache=list(CACHE.values()))
    print("Step 2 DONE")


if __name__ == "__main__":
    main()
