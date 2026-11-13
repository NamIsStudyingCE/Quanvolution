"""
Export Trained WideGRQuanvNet v2 Checkpoints & Real MedMNIST Test Set for Pure Rust Native Inference
====================================================================================================
Exports:
  1. weights/wide_v2_pneumoniamnist_s42_rust.bin
     Folded BatchNorm+Conv2d weights in flat little-endian f32 binary format (Magic: b"MQRW").
  2. data/processed/pneumoniamnist_test_rust_verify.bin
     All 624 real PneumoniaMNIST test images (28x28 uint8), labels (uint8),
     plus PyTorch reference logits (f32) for:
       - Mode 0: INT8 Shift-LUT (256 KiB, p >> 4)
       - Mode 1: INT8 Grid-LUT  (256 KiB, round(p*15/255))
       - Mode 2: Interp4D SIMD  (1,024 KiB, FP32)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import rust_qlut
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from models.gr_quanv import _build_axial_u8, _build_dilated_u8
from models.gr_quanv_wide import WideGRQuanvNet
from run_pillar2_expansion import load


def fold_conv_bn(conv_w: torch.Tensor, conv_b: torch.Tensor | None,
                 bn_w: torch.Tensor, bn_b: torch.Tensor,
                 bn_mean: torch.Tensor, bn_var: torch.Tensor, eps: float = 1e-5):
    """Mathematically exact BatchNorm2d folding into preceding Conv2d."""
    inv_std = bn_w / torch.sqrt(bn_var + eps)
    w_folded = conv_w * inv_std.view(-1, 1, 1, 1)
    if conv_b is None:
        conv_b = torch.zeros_like(bn_mean)
    b_folded = (conv_b - bn_mean) * inv_std + bn_b
    return w_folded.detach().cpu().numpy().astype(np.float32), b_folded.detach().cpu().numpy().astype(np.float32)


def bn_to_affine(bn_w: torch.Tensor, bn_b: torch.Tensor,
                 bn_mean: torch.Tensor, bn_var: torch.Tensor, eps: float = 1e-5):
    """Convert standalone pre-activation BatchNorm2d into per-channel (scale, shift)."""
    scale = bn_w / torch.sqrt(bn_var + eps)
    shift = bn_b - bn_mean * scale
    return scale.detach().cpu().numpy().astype(np.float32), shift.detach().cpu().numpy().astype(np.float32)


def extract_qlut_mode(imgs_u8: np.ndarray, mode: str) -> torch.Tensor:
    imgs_u8 = np.ascontiguousarray(imgs_u8, dtype=np.uint8)
    N = imgs_u8.shape[0]
    inputs = [
        imgs_u8,
        _build_dilated_u8(imgs_u8, 2),
        _build_dilated_u8(imgs_u8, 3),
        _build_axial_u8(imgs_u8),
    ]
    if mode == "interp4d":
        scales = []
        for u in inputs:
            s = np.empty((N, 4, 14, 14), dtype=np.float32)
            rust_qlut.forward_interp4d_nchw(u, s, parallel=(N >= 64))
            scales.append(s)
        return torch.from_numpy(np.concatenate(scales, axis=1))
    else:
        use_shift = (mode == "int8_shift")
        scales = []
        for u in inputs:
            s = np.empty((N, 4, 14, 14), dtype=np.int8)
            rust_qlut.forward_int8_nchw(u, s, use_shift=use_shift, parallel=(N >= 64))
            scales.append(s)
        return torch.from_numpy(np.concatenate(scales, axis=1)).to(torch.float32) / 127.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(PROJECT_ROOT / "weights" / "pillar2_wide" / "wide_v2_interp4d_pneumoniamnist_full_grquanv_s42.pt"),
    )
    parser.add_argument(
        "--npz",
        default=str(PROJECT_ROOT / "data" / "pneumoniamnist.npz"),
    )
    parser.add_argument(
        "--out-weights",
        default=str(PROJECT_ROOT / "weights" / "wide_v2_pneumoniamnist_s42_rust.bin"),
    )
    parser.add_argument(
        "--out-verify",
        default=str(PROJECT_ROOT / "data" / "processed" / "pneumoniamnist_test_rust_verify.bin"),
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    model = WideGRQuanvNet(num_classes=2)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    n_params_raw = sum(p.numel() for p in model.parameters())
    print(f"[Loaded Checkpoint] {ckpt_path.name} ({n_params_raw:,} trainable params)")

    tensors_ordered = []

    # 1. c_stem + norm_c
    w, b = fold_conv_bn(sd["c_stem.weight"], sd["c_stem.bias"],
                        sd["norm_c.weight"], sd["norm_c.bias"],
                        sd["norm_c.running_mean"], sd["norm_c.running_var"])
    tensors_ordered.extend([w, b])

    # 2. q_mixer: bn0, conv1+bn2, conv4+bn5, conv7+norm_q
    s, sh = bn_to_affine(sd["q_mixer.0.weight"], sd["q_mixer.0.bias"],
                         sd["q_mixer.0.running_mean"], sd["q_mixer.0.running_var"])
    tensors_ordered.extend([s, sh])

    w, b = fold_conv_bn(sd["q_mixer.1.weight"], sd["q_mixer.1.bias"],
                        sd["q_mixer.2.weight"], sd["q_mixer.2.bias"],
                        sd["q_mixer.2.running_mean"], sd["q_mixer.2.running_var"])
    tensors_ordered.extend([w, b])

    w, b = fold_conv_bn(sd["q_mixer.4.weight"], sd["q_mixer.4.bias"],
                        sd["q_mixer.5.weight"], sd["q_mixer.5.bias"],
                        sd["q_mixer.5.running_mean"], sd["q_mixer.5.running_var"])
    tensors_ordered.extend([w, b])

    w, b = fold_conv_bn(sd["q_mixer.7.weight"], sd["q_mixer.7.bias"],
                        sd["norm_q.weight"], sd["norm_q.bias"],
                        sd["norm_q.running_mean"], sd["norm_q.running_var"])
    tensors_ordered.extend([w, b])

    # 3. gate: bn0, conv1, conv3
    s, sh = bn_to_affine(sd["gate.0.weight"], sd["gate.0.bias"],
                         sd["gate.0.running_mean"], sd["gate.0.running_var"])
    tensors_ordered.extend([s, sh])
    tensors_ordered.extend([
        sd["gate.1.weight"].detach().cpu().numpy().astype(np.float32),
        sd["gate.1.bias"].detach().cpu().numpy().astype(np.float32),
        sd["gate.3.weight"].detach().cpu().numpy().astype(np.float32),
        sd["gate.3.bias"].detach().cpu().numpy().astype(np.float32),
    ])

    # 4. head: pre_bn0, stem (c+bn), block1 (c1+b1, c2+b2), block2 (c1+b1, c2+b2, skip+skip_bn), fc
    s, sh = bn_to_affine(sd["head.pre.0.weight"], sd["head.pre.0.bias"],
                         sd["head.pre.0.running_mean"], sd["head.pre.0.running_var"])
    tensors_ordered.extend([s, sh])

    w, b = fold_conv_bn(sd["head.stem.0.weight"], None,
                        sd["head.stem.1.weight"], sd["head.stem.1.bias"],
                        sd["head.stem.1.running_mean"], sd["head.stem.1.running_var"])
    tensors_ordered.extend([w, b])

    for blk in ("block1", "block2"):
        w, b = fold_conv_bn(sd[f"head.{blk}.c1.weight"], None,
                            sd[f"head.{blk}.b1.weight"], sd[f"head.{blk}.b1.bias"],
                            sd[f"head.{blk}.b1.running_mean"], sd[f"head.{blk}.b1.running_var"])
        tensors_ordered.extend([w, b])
        w, b = fold_conv_bn(sd[f"head.{blk}.c2.weight"], None,
                            sd[f"head.{blk}.b2.weight"], sd[f"head.{blk}.b2.bias"],
                            sd[f"head.{blk}.b2.running_mean"], sd[f"head.{blk}.b2.running_var"])
        tensors_ordered.extend([w, b])

    w, b = fold_conv_bn(sd["head.block2.skip.0.weight"], None,
                        sd["head.block2.skip.1.weight"], sd["head.block2.skip.1.bias"],
                        sd["head.block2.skip.1.running_mean"], sd["head.block2.skip.1.running_var"])
    tensors_ordered.extend([w, b])

    tensors_ordered.extend([
        sd["head.fc.weight"].detach().cpu().numpy().astype(np.float32),
        sd["head.fc.bias"].detach().cpu().numpy().astype(np.float32),
    ])

    total_f32 = sum(arr.size for arr in tensors_ordered)
    out_w_path = Path(args.out_weights)
    out_w_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_w_path, "wb") as f:
        f.write(b"MQRW")
        f.write(struct.pack("<III", 1, 2, total_f32))  # version=1, num_classes=2, total_floats
        for arr in tensors_ordered:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    print(f"[Saved Folded Weights] {out_w_path} ({out_w_path.stat().st_size:,} bytes, {total_f32:,} f32 values)")

    # Export 624 real PneumoniaMNIST test images + labels + PyTorch reference logits
    D = load(args.npz)
    te_u8 = np.ascontiguousarray(D["te"], dtype=np.uint8)
    y_u8 = np.ascontiguousarray(D["yte"], dtype=np.uint8)
    N = len(te_u8)
    x_t = (torch.from_numpy(te_u8).float().unsqueeze(1) / 255.0 - 0.5) / 0.5

    logits_modes = {}
    with torch.no_grad():
        for mode in ("int8_shift", "int8_grid", "interp4d"):
            q_t = extract_qlut_mode(te_u8, mode)
            lg = model(x_t, q_t).cpu().numpy().astype(np.float32)
            acc = float(np.mean(lg.argmax(axis=1) == y_u8) * 100.0)
            logits_modes[mode] = lg
            print(f"  [PyTorch Reference | {mode:<10}] N={N}, Clean Test ACC = {acc:.2f}%")

    out_v_path = Path(args.out_verify)
    out_v_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_v_path, "wb") as f:
        f.write(b"MQRV")
        f.write(struct.pack("<IIII", 1, N, 28, 28))
        f.write(te_u8.tobytes())
        f.write(y_u8.tobytes())
        for mode in ("int8_shift", "int8_grid", "interp4d"):
            f.write(np.ascontiguousarray(logits_modes[mode], dtype="<f4").tobytes())

    print(f"[Saved Test Verify Bin] {out_v_path} ({out_v_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
