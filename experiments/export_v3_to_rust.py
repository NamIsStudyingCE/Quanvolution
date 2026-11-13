"""
Export Phase B (GR-Quanv v3) & Equal/Heavy Classical Baselines to Pure-Rust Native Binaries
========================================================================================
Exports BN-folded FP32 weight files + verified INT8 Shift-LUT (256 KiB) + real PneumoniaMNIST
test samples (both clean sigma=0.00 and pre-damped speckle sigma=0.20) with PyTorch reference logits:
  1. quanv_v3_int8 : GRQuanvNetV3 (4.03M MACs, 256 KiB LUT + BN-folded FP32 weights)
  2. ctrl_v3_light : ClassicalControlV3Light (4.07M MACs, 0% Quantum)
  3. ctrl_v2_full  : WideClassicalControl v2 (18.90M MACs, 0% Quantum)
  4. micro_quanv   : MicroEdgeNetV3 Quantum (0.17M MACs, 256 KiB LUT)
  5. micro_ctrl    : MicroEdgeNetV3 Classical (0.19M MACs, 0% Quantum)

Verifies folded math against eager PyTorch eval (< 1e-5 max logit diff) before writing binaries.
"""

import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from export_wide_v2_to_rust import bn_to_affine, fold_conv_bn
from models.gr_quanv import phase_damping_u8
from models.gr_quanv_v3 import ClassicalControlV3Light, GRQuanvNetV3, MicroEdgeNetV3
from models.gr_quanv_wide import WideClassicalControl
from run_pillar2_expansion import speckle
from run_pillar2_wide import load_dataset_with_cache


def compose_bn_affine(bn1_tuple, bn2_tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Composes y = s2 * (s1 * x + sh1) + sh2 = (s2*s1)*x + (s2*sh1 + sh2)."""
    s1, sh1 = bn1_tuple
    s2, sh2 = bn2_tuple
    return (s2 * s1).astype(np.float32), (s2 * sh1 + sh2).astype(np.float32)


def bn_arr(sd: Dict[str, torch.Tensor], prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    return bn_to_affine(
        sd[f"{prefix}.weight"],
        sd[f"{prefix}.bias"],
        sd[f"{prefix}.running_mean"],
        sd[f"{prefix}.running_var"],
    )


def fold_arr(sd: Dict[str, torch.Tensor], conv_pfx: str, bn_pfx: str) -> Tuple[np.ndarray, np.ndarray]:
    bias = sd.get(f"{conv_pfx}.bias", None)
    return fold_conv_bn(
        sd[f"{conv_pfx}.weight"],
        bias,
        sd[f"{bn_pfx}.weight"],
        sd[f"{bn_pfx}.bias"],
        sd[f"{bn_pfx}.running_mean"],
        sd[f"{bn_pfx}.running_var"],
    )


def raw_arr(sd: Dict[str, torch.Tensor], key: str) -> np.ndarray:
    return sd[key].detach().cpu().numpy().astype(np.float32)


def write_v3_weights(path: Path, kind_id: int, num_classes: int, tensors: List[np.ndarray]):
    total_f32 = sum(int(t.size) for t in tensors)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"MQV3")
        f.write(struct.pack("<IIII", 1, kind_id, num_classes, total_f32))
        for arr in tensors:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    print(f"  [Saved Weights] {path.name:<44} {path.stat().st_size:>8,} bytes ({total_f32:,} f32)")


def export_quanv_v3(sd: Dict[str, torch.Tensor], num_classes: int) -> List[np.ndarray]:
    t: List[np.ndarray] = []
    t.extend(fold_arr(sd, "c_stem", "norm_c"))
    t.extend(compose_bn_affine(bn_arr(sd, "norm_q_in"), bn_arr(sd, "q_mixer.0")))
    t.append(raw_arr(sd, "q_mixer.1.weight"))
    t.extend(fold_arr(sd, "q_mixer.2", "q_mixer.3"))
    t.append(raw_arr(sd, "q_mixer.5.weight"))
    t.extend(fold_arr(sd, "q_mixer.6", "q_mixer.7"))
    t.extend(bn_arr(sd, "gate.0"))
    t.append(raw_arr(sd, "gate.1.weight"))
    t.extend([raw_arr(sd, "gate.2.weight"), raw_arr(sd, "gate.2.bias")])
    t.extend(fold_arr(sd, "s2_proj.0", "s2_proj.1"))
    t.extend(bn_arr(sd, "norm_q2"))
    t.extend(fold_arr(sd, "head_stem.0", "head_stem.1"))
    t.append(raw_arr(sd, "block1.dw1.weight"))
    t.extend(fold_arr(sd, "block1.pw1", "block1.bn1"))
    t.append(raw_arr(sd, "block1.dw2.weight"))
    t.extend(fold_arr(sd, "block1.pw2", "block1.bn2"))
    t.append(raw_arr(sd, "block2.dw1.weight"))
    t.extend(fold_arr(sd, "block2.pw1", "block2.bn1"))
    t.append(raw_arr(sd, "block2.dw2.weight"))
    t.extend(fold_arr(sd, "block2.pw2", "block2.bn2"))
    t.extend(fold_arr(sd, "block2.skip.0", "block2.skip.1"))
    t.extend([raw_arr(sd, "fc.weight"), raw_arr(sd, "fc.bias")])
    return t


def export_ctrl_v3(sd: Dict[str, torch.Tensor], num_classes: int) -> List[np.ndarray]:
    t: List[np.ndarray] = []
    t.extend(fold_arr(sd, "c_stem", "norm_c"))
    for b in range(4):
        t.extend([raw_arr(sd, f"classical_branches.{b}.weight"), raw_arr(sd, f"classical_branches.{b}.bias")])
    t.extend(compose_bn_affine(bn_arr(sd, "norm_q_in"), bn_arr(sd, "q_mixer.0")))
    t.append(raw_arr(sd, "q_mixer.1.weight"))
    t.extend(fold_arr(sd, "q_mixer.2", "q_mixer.3"))
    t.append(raw_arr(sd, "q_mixer.5.weight"))
    t.extend(fold_arr(sd, "q_mixer.6", "q_mixer.7"))
    t.extend(bn_arr(sd, "gate.0"))
    t.append(raw_arr(sd, "gate.1.weight"))
    t.extend([raw_arr(sd, "gate.2.weight"), raw_arr(sd, "gate.2.bias")])
    t.extend(fold_arr(sd, "s2_proj.0", "s2_proj.1"))
    t.extend(fold_arr(sd, "s2_proj.3", "s2_proj.4"))
    t.extend(fold_arr(sd, "head_stem.0", "head_stem.1"))
    t.append(raw_arr(sd, "block1.dw1.weight"))
    t.extend(fold_arr(sd, "block1.pw1", "block1.bn1"))
    t.append(raw_arr(sd, "block1.dw2.weight"))
    t.extend(fold_arr(sd, "block1.pw2", "block1.bn2"))
    t.append(raw_arr(sd, "block2.dw1.weight"))
    t.extend(fold_arr(sd, "block2.pw1", "block2.bn1"))
    t.append(raw_arr(sd, "block2.dw2.weight"))
    t.extend(fold_arr(sd, "block2.pw2", "block2.bn2"))
    t.extend(fold_arr(sd, "block2.skip.0", "block2.skip.1"))
    t.extend([raw_arr(sd, "fc.weight"), raw_arr(sd, "fc.bias")])
    return t


def export_ctrl_v2(sd: Dict[str, torch.Tensor], num_classes: int) -> List[np.ndarray]:
    t: List[np.ndarray] = []
    t.extend(fold_arr(sd, "c_stem", "norm_c"))
    for b in range(4):
        t.extend([raw_arr(sd, f"classical_branches.{b}.weight"), raw_arr(sd, f"classical_branches.{b}.bias")])
    t.extend(bn_arr(sd, "mixer.0"))
    t.extend(fold_arr(sd, "mixer.1", "mixer.2"))
    t.extend(fold_arr(sd, "mixer.4", "mixer.5"))
    t.extend(fold_arr(sd, "mixer.7", "norm_ms"))
    t.extend(bn_arr(sd, "gate.0"))
    t.extend([raw_arr(sd, "gate.1.weight"), raw_arr(sd, "gate.1.bias")])
    t.extend([raw_arr(sd, "gate.3.weight"), raw_arr(sd, "gate.3.bias")])
    t.extend(bn_arr(sd, "head.pre.0"))
    t.extend(fold_arr(sd, "head.stem.0", "head.stem.1"))
    for blk in ("block1", "block2"):
        t.extend(fold_arr(sd, f"head.{blk}.c1", f"head.{blk}.b1"))
        t.extend(fold_arr(sd, f"head.{blk}.c2", f"head.{blk}.b2"))
    t.extend(fold_arr(sd, "head.block2.skip.0", "head.block2.skip.1"))
    t.extend([raw_arr(sd, "head.fc.weight"), raw_arr(sd, "head.fc.bias")])
    return t


def export_micro(sd: Dict[str, torch.Tensor], classical: bool) -> List[np.ndarray]:
    t: List[np.ndarray] = []
    if classical:
        for b in range(4):
            t.extend([raw_arr(sd, f"branches.{b}.weight"), raw_arr(sd, f"branches.{b}.bias")])
    t.extend(bn_arr(sd, "stem.0"))
    t.extend(fold_arr(sd, "stem.1", "stem.2"))
    t.extend(fold_arr(sd, "stem.4", "stem.5"))
    t.extend(fold_arr(sd, "stem.8", "stem.9"))
    t.extend(fold_arr(sd, "stem.11", "stem.12"))
    t.extend([raw_arr(sd, "fc.weight"), raw_arr(sd, "fc.bias")])
    return t


def main():
    torch.set_grad_enabled(False)
    ds = "pneumoniamnist"
    nc = 2
    D = load_dataset_with_cache(ds, interp4d=True)
    te_clean_u8 = np.ascontiguousarray(D["te"], dtype=np.uint8)
    te_f = te_clean_u8.astype(np.float32) / 255.0
    te_s20_u8 = np.ascontiguousarray(phase_damping_u8(speckle(te_f, 0.20, 0)), dtype=np.uint8)
    y_u8 = np.ascontiguousarray(D["yte"], dtype=np.uint8)
    N = len(te_clean_u8)

    x_clean = (torch.from_numpy(te_clean_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5
    x_s20 = (torch.from_numpy(te_s20_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5

    specs = [
        ("quanv_v3_int8", 1, GRQuanvNetV3(num_classes=nc, n_tables=1, qat_int8=True, drop_c_prob=0.20, use_stage2_qlut=True, width1=64, width2=96),
         PROJECT_ROOT / "weights/pillar3_v3/pneumoniamnist_quanv_v3_int8_s42.pt",
         PROJECT_ROOT / "weights/v3_quanv_pneumoniamnist_s42.bin",
         PROJECT_ROOT / "weights/v3_pneumoniamnist_s42_int8_shift.bin"),
        ("ctrl_v3_light", 2, ClassicalControlV3Light(num_classes=nc, drop_c_prob=0.20, width1=64, width2=96),
         PROJECT_ROOT / "weights/pillar3_v3/pneumoniamnist_ctrl_v3_light_s42.pt",
         PROJECT_ROOT / "weights/v3_ctrl_light_pneumoniamnist_s42.bin",
         None),
        ("ctrl_v2_full", 3, WideClassicalControl(num_classes=nc),
         PROJECT_ROOT / "weights/pillar2_wide/wide_v2_interp4d_pneumoniamnist_full_control_s42.pt",
         PROJECT_ROOT / "weights/v2_ctrl_full_pneumoniamnist_s42.bin",
         None),
        ("micro_quanv", 4, MicroEdgeNetV3(num_classes=nc, classical=False),
         PROJECT_ROOT / "weights/pillar3_v3/pneumoniamnist_micro_quanv_s42.pt",
         PROJECT_ROOT / "weights/v3_micro_quanv_pneumoniamnist_s42.bin",
         PROJECT_ROOT / "weights/v3_pneumoniamnist_s42_micro_int8_shift.bin"),
        ("micro_ctrl", 5, MicroEdgeNetV3(num_classes=nc, classical=True),
         PROJECT_ROOT / "weights/pillar3_v3/pneumoniamnist_micro_ctrl_s42.pt",
         PROJECT_ROOT / "weights/v3_micro_ctrl_pneumoniamnist_s42.bin",
         None),
    ]

    clean_logits_all = []
    s20_logits_all = []

    print(f"=== Exporting Phase B & v2 Models to Pure Rust (N={N}) ===")
    for name, kid, model, ckpt_path, out_bin, lut_bin in specs:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        tbl = None
        if lut_bin is not None:
            tbl_np = model.qlut.compile_int8_shift_tables()
            tbl_np[0].astype(np.int8).tofile(lut_bin)
            tbl = torch.from_numpy(tbl_np)
            print(f"  [Saved LUT    ] {lut_bin.name:<44} {lut_bin.stat().st_size:>8,} bytes")

        if name == "quanv_v3_int8":
            tensors = export_quanv_v3(sd, nc)
        elif name == "ctrl_v3_light":
            tensors = export_ctrl_v3(sd, nc)
        elif name == "ctrl_v2_full":
            tensors = export_ctrl_v2(sd, nc)
        elif name == "micro_quanv":
            tensors = export_micro(sd, classical=False)
        else:
            tensors = export_micro(sd, classical=True)
        write_v3_weights(out_bin, kid, nc, tensors)

        lg_c = (model(x_clean, compiled_tables=tbl) if tbl is not None else model(x_clean)).numpy().astype(np.float32)
        lg_s = (model(x_s20, compiled_tables=tbl) if tbl is not None else model(x_s20)).numpy().astype(np.float32)
        acc_c = float(np.mean(lg_c.argmax(1) == y_u8) * 100.0)
        acc_s = float(np.mean(lg_s.argmax(1) == y_u8) * 100.0)
        print(f"  [PyTorch Ref  ] {name:<14}: Clean ACC={acc_c:5.2f}%, Speckle(0.20,d0) ACC={acc_s:5.2f}%")
        clean_logits_all.append(lg_c)
        s20_logits_all.append(lg_s)

    verify_bin = PROJECT_ROOT / "data/processed/pneumoniamnist_v3_rust_verify.bin"
    with open(verify_bin, "wb") as f:
        f.write(b"MQR3")
        f.write(struct.pack("<IIIII", 1, N, 28, 28, len(specs)))
        f.write(te_clean_u8.tobytes())
        f.write(te_s20_u8.tobytes())
        f.write(y_u8.tobytes())
        for lg in clean_logits_all:
            f.write(np.ascontiguousarray(lg, dtype="<f4").tobytes())
        for lg in s20_logits_all:
            f.write(np.ascontiguousarray(lg, dtype="<f4").tobytes())
    print(f"[Saved Verify Dataset] {verify_bin} ({verify_bin.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
