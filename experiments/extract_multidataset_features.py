"""
BƯỚC 2: Tiền trích xuất đặc trưng Rust Q-LUT đa tỷ lệ (d=1,2,3,axial -> 16 kênh) cho 3 tập MedMNIST.
- Chế độ mặc định (INT8 4-bit shift):
    Cache: data/processed/{breast,pneumonia}_qlut.npz
- Chế độ `--interp4d` (Nội suy đa tuyến tính 4D SIMD FP32 - Cosine 0.999993):
    Cache: data/processed/{breast,pneumonia}_qlut_interp.npz
Kèm kiểm tra tính đúng đắn (roundtrip) và ghi thời gian vào reports/pillar2_expansion_extraction.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import pillar2_state as st
from models.gr_quanv import extract_multiscale_qlut

SETS = {"breast": "data/breastmnist.npz", "pneumonia": "data/pneumoniamnist.npz", "oct": "data/octmnist.npz"}
CACHE_INT8 = {"breast": "data/processed/breast_qlut.npz", "pneumonia": "data/processed/pneumonia_qlut.npz"}
CACHE_INTERP = {"breast": "data/processed/breast_qlut_interp.npz", "pneumonia": "data/processed/pneumonia_qlut_interp.npz"}


def extract_caches(interp4d: bool = False, include_oct_bench: bool = True):
    cache_map = CACHE_INTERP if interp4d else CACHE_INT8
    mode_tag = "interp4d_fp32" if interp4d else "int8_shift"
    report = {}
    sets_to_run = SETS if include_oct_bench else {k: SETS[k] for k in cache_map}
    for name, path in sets_to_run.items():
        d = np.load(path)
        out, rep = {}, {}
        for split, key in (("train", "train_images"), ("val", "val_images"), ("test", "test_images")):
            u8 = np.ascontiguousarray(d[key], dtype=np.uint8)
            extract_multiscale_qlut(u8[:min(64, len(u8))], include_axial=True, interp4d=interp4d)  # warm-up
            t0 = time.perf_counter()
            q, _ = extract_multiscale_qlut(u8, include_axial=True, interp4d=interp4d)
            ms = (time.perf_counter() - t0) * 1000
            if interp4d:
                q_arr = q.numpy().astype(np.float32)
            else:
                q_arr = np.rint(q.numpy() * 127.0).astype(np.int8)
            rep[split] = dict(
                mode=mode_tag,
                n=int(len(u8)),
                ms=ms,
                us_per_img=ms * 1000 / len(u8),
                shape=list(q_arr.shape),
                dtype=str(q_arr.dtype),
            )
            out[f"q_{split}"] = q_arr
        if name in cache_map:
            os.makedirs(os.path.dirname(cache_map[name]), exist_ok=True)
            np.savez_compressed(cache_map[name], **out)
            chk, _ = extract_multiscale_qlut(
                np.ascontiguousarray(d["test_images"], dtype=np.uint8),
                include_axial=True,
                interp4d=interp4d,
            )
            loaded = np.load(cache_map[name])["q_test"]
            if interp4d:
                ok = np.allclose(loaded, chk.numpy(), atol=1e-6)
            else:
                ok = np.array_equal(loaded, np.rint(chk.numpy() * 127.0).astype(np.int8))
            rep["cache_roundtrip_ok"] = bool(ok)
            assert ok, f"cache mismatch {name} ({mode_tag})"
        report[name] = rep
        print(f"[{mode_tag}] {name}:", {k: (round(v["ms"], 2) if isinstance(v, dict) else v) for k, v in rep.items()}, flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interp4d", action="store_true", help="Extract continuous 4D multilinear interpolated Q-LUT caches")
    p.add_argument("--skip-oct", action="store_true", help="Skip OCT full benchmark timing during quick cache generation")
    a = p.parse_args()

    json_path = "reports/pillar2_expansion_extraction.json"
    existing = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    if a.interp4d:
        rep = extract_caches(interp4d=True, include_oct_bench=not a.skip_oct)
        existing["interp4d"] = rep
        st.atomic_dump_json(existing, json_path)
        st.mark("step2_interp4d_extraction", True, cache=list(CACHE_INTERP.values()))
        print("Step 2 (Interp4D caches) DONE")
    else:
        if st.is_done("step2_feature_extraction") and all(os.path.exists(p) for p in CACHE_INT8.values()):
            print("Step 2 (INT8) already completed -> skip")
            return
        rep = extract_caches(interp4d=False, include_oct_bench=not a.skip_oct)
        existing.update(rep)
        st.atomic_dump_json(existing, json_path)
        st.mark("step2_feature_extraction", True, cache=list(CACHE_INT8.values()))
        print("Step 2 (INT8 caches) DONE")


if __name__ == "__main__":
    main()
