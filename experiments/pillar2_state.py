"""
Bộ ghi trạng thái (checkpoint) cho chuỗi thực nghiệm mở rộng Cột trụ 2.
File trạng thái: experiments/.state_pillar2_expansion.json
Mọi ghi đều atomic (ghi file tạm rồi os.replace) để không bao giờ hỏng file khi bị ngắt giữa chừng.
"""
from __future__ import annotations
import json
import os
import time

STATE_PATH = "experiments/.state_pillar2_expansion.json"
STEPS = {
    "step1_architecture": "src/models/gr_quanv_resblock.py",
    "step2_feature_extraction": "experiments/extract_multidataset_features.py",
    "step3_multidataset_training": "experiments/run_pillar2_expansion.py",
    "step4_speckle_stress": "experiments/run_pillar2_expansion.py --stage speckle",
    "step5_report_figures": "experiments/plot_pillar2_expansion.py",
}


def _default() -> dict:
    return {
        "steps": {k: {"completed": False, "artifact": v} for k, v in STEPS.items()},
        "notes": [],
    }


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _default()


def _atomic_write(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark(step: str, completed: bool = True, **extra) -> None:
    st = load_state()
    st["steps"].setdefault(step, {})
    st["steps"][step]["completed"] = completed
    st["steps"][step]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    st["steps"][step].update(extra)
    _atomic_write(st, STATE_PATH)


def note(msg: str) -> None:
    st = load_state()
    st["notes"].append(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}")
    _atomic_write(st, STATE_PATH)


def is_done(step: str) -> bool:
    return bool(load_state()["steps"].get(step, {}).get("completed", False))


def atomic_dump_json(obj: dict, path: str) -> None:
    _atomic_write(obj, path)
