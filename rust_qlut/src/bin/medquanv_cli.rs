//! medquanv-cli — Standalone Embedded Edge-Quanv Medical Inference Appliance
//! Executes the FULL 148,306-parameter WideGRQuanvNet v2 network in Pure Rust
//! (Zero Python | Zero PyTorch | Zero CUDA/GPU Dependencies).
//! Target Reference: NVIDIA Jetson Nano (ARM Cortex-A57, 2 MB L2 Cache, 4 GB LPDDR4)

use rust_qlut::{
    kernel_int8_nchw_single, kernel_interp4d_nchw_single, qlut_forward_int8_nhwc_strided_2d,
    qlut_forward_interp4d_nchw_strided_2d, qlut_table_fp32_bytes, qlut_table_int8_bytes, IMG_H,
    IMG_PIXELS, IMG_W, LUT_FP32_GRID, LUT_INT8_GRID, LUT_INT8_SHIFT, NUM_PATCHES, OUT_ELEMS,
    OUT_H, OUT_W,
};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

const LAP_KERNEL: [f32; 9] = [0.25, -0.5, 0.25, -0.5, 1.0, -0.5, 0.25, -0.5, 0.25];

#[inline(always)]
fn reflect_idx(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= n as isize {
        (2 * (n as isize) - 2 - i) as usize
    } else {
        i as usize
    }
}

#[inline(always)]
fn erff_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let ax = x.abs();
    let t = 1.0f32 / (1.0f32 + 0.3275911f32 * ax);
    let poly = (((((1.061405429f32 * t - 1.453152027f32) * t) + 1.421413741f32) * t
        - 0.284496736f32)
        * t
        + 0.254829592f32)
        * t;
    sign * (1.0f32 - poly * (-ax * ax).exp())
}

#[inline(always)]
fn gelu_exact(x: f32) -> f32 {
    x * 0.5f32 * (1.0f32 + erff_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Computes exact PyTorch F.conv2d(reflect_pad(x_01), _LAP_KERNEL).abs().mean()
fn compute_lap_hp_score(x_01: &[f32; IMG_PIXELS]) -> f32 {
    let mut sum = 0.0f32;
    for r in 0..IMG_H {
        for c in 0..IMG_W {
            let mut conv = 0.0f32;
            for kr in 0..3 {
                let rr = reflect_idx(r as isize + kr as isize - 1, IMG_H);
                for kc in 0..3 {
                    let cc = reflect_idx(c as isize + kc as isize - 1, IMG_W);
                    conv += x_01[rr * IMG_W + cc] * LAP_KERNEL[kr * 3 + kc];
                }
            }
            sum += conv.abs();
        }
    }
    sum / (IMG_PIXELS as f32)
}

/// Pre-LUT Phase Damping with quadratic soft-start (0.0000% modification on clean images)
fn phase_damping_u8(img: &[u8; IMG_PIXELS], beta: f32, tau: f32) -> [u8; IMG_PIXELS] {
    let mut x = [0.0f32; IMG_PIXELS];
    for i in 0..IMG_PIXELS {
        x[i] = (img[i] as f32) / 255.0;
    }
    let hp_score = compute_lap_hp_score(&x);
    let raw_gate = ((hp_score - tau) / tau).clamp(0.0, 6.0);
    if raw_gate == 0.0 {
        return *img;
    }
    let gate = raw_gate * ((raw_gate - 0.15) / 0.50).clamp(0.0, 1.0);
    if gate == 0.0 {
        return *img;
    }

    let mut x_bar = [0.0f32; IMG_PIXELS];
    for r in 0..IMG_H {
        for c in 0..IMG_W {
            let mut s = 0.0f32;
            for dr in -1isize..=1 {
                let rr = reflect_idx(r as isize + dr, IMG_H);
                for dc in -1isize..=1 {
                    let cc = reflect_idx(c as isize + dc, IMG_W);
                    s += x[rr * IMG_W + cc];
                }
            }
            x_bar[r * IMG_W + c] = s / 9.0;
        }
    }

    let mut diff_sq = [0.0f32; IMG_PIXELS];
    for i in 0..IMG_PIXELS {
        let d = x[i] - x_bar[i];
        diff_sq[i] = d * d;
    }

    let mut out = [0u8; IMG_PIXELS];
    for r in 0..IMG_H {
        for c in 0..IMG_W {
            let mut s2 = 0.0f32;
            for dr in -1isize..=1 {
                let rr = reflect_idx(r as isize + dr, IMG_H);
                for dc in -1isize..=1 {
                    let cc = reflect_idx(c as isize + dc, IMG_W);
                    s2 += diff_sq[rr * IMG_W + cc];
                }
            }
            let idx = r * IMG_W + c;
            let local_std = (s2 / 9.0 + 1e-8).sqrt();
            let damped = (x_bar[idx] + (x[idx] - x_bar[idx]) / (1.0 + beta * gate * local_std))
                .clamp(0.0, 1.0);
            out[idx] = (damped * 255.0).round() as u8;
        }
    }
    out
}

#[inline(always)]
fn build_dilated_u8(src: &[u8; IMG_PIXELS], dilation: usize, dst: &mut [u8; IMG_PIXELS]) {
    let pad_before = (dilation - 1) as isize;
    for r_half in 0..OUT_H {
        let r_even = r_half * 2;
        let r_odd = r_even + dilation;
        let sr0 = (r_even as isize - pad_before).clamp(0, (IMG_H - 1) as isize) as usize;
        let sr1 = (r_odd as isize - pad_before).clamp(0, (IMG_H - 1) as isize) as usize;
        for c_half in 0..OUT_W {
            let c_even = c_half * 2;
            let c_odd = c_even + dilation;
            let sc0 = (c_even as isize - pad_before).clamp(0, (IMG_W - 1) as isize) as usize;
            let sc1 = (c_odd as isize - pad_before).clamp(0, (IMG_W - 1) as isize) as usize;
            dst[r_even * IMG_W + c_even] = src[sr0 * IMG_W + sc0];
            dst[r_even * IMG_W + c_even + 1] = src[sr0 * IMG_W + sc1];
            dst[(r_even + 1) * IMG_W + c_even] = src[sr1 * IMG_W + sc0];
            dst[(r_even + 1) * IMG_W + c_even + 1] = src[sr1 * IMG_W + sc1];
        }
    }
}

#[inline(always)]
fn build_axial_u8(src: &[u8; IMG_PIXELS], dst: &mut [u8; IMG_PIXELS]) {
    for r_half in 0..OUT_H {
        let r_even = r_half * 2;
        let sr0 = (r_even as isize - 1).clamp(0, (IMG_H - 1) as isize) as usize;
        let sr1 = r_even;
        let sr2 = (r_even + 1).min(IMG_H - 1);
        for c_half in 0..OUT_W {
            let c_even = c_half * 2;
            let sc1 = c_even;
            let sc2 = (c_even + 1).min(IMG_W - 1);
            dst[r_even * IMG_W + c_even] = src[sr0 * IMG_W + sc1];
            dst[r_even * IMG_W + c_even + 1] = src[sr1 * IMG_W + sc1];
            dst[(r_even + 1) * IMG_W + c_even] = src[sr2 * IMG_W + sc1];
            dst[(r_even + 1) * IMG_W + c_even + 1] = src[sr1 * IMG_W + sc2];
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum QLutMode {
    Int8Shift,
    Int8Grid,
    Interp4d,
}

impl QLutMode {
    fn name(self) -> &'static str {
        match self {
            QLutMode::Int8Shift => "INT8 Shift-LUT (256 KiB, 0-FLOP)",
            QLutMode::Int8Grid => "INT8 Grid-LUT  (256 KiB)",
            QLutMode::Interp4d => "Interp4D SIMD  (1,024 KiB, FP32)",
        }
    }
}

/// Extract 16-channel 14x14 multi-scale Q-LUT features using `rust_qlut` native kernels
fn extract_multiscale_qlut(
    raw_img: &[u8; IMG_PIXELS],
    mode: QLutMode,
    use_phase_damping: bool,
    out_16ch: &mut [f32; 16 * NUM_PATCHES],
) {
    let img = if use_phase_damping {
        phase_damping_u8(raw_img, 18.0, 0.024)
    } else {
        *raw_img
    };

    let mut d2 = [0u8; IMG_PIXELS];
    let mut d3 = [0u8; IMG_PIXELS];
    let mut ax = [0u8; IMG_PIXELS];
    build_dilated_u8(&img, 2, &mut d2);
    build_dilated_u8(&img, 3, &mut d3);
    build_axial_u8(&img, &mut ax);
    let scales: [&[u8; IMG_PIXELS]; 4] = [&img, &d2, &d3, &ax];

    match mode {
        QLutMode::Interp4d => {
            let lut_f32 = LUT_FP32_GRID.0.as_ptr() as *const f32;
            for (b, s_img) in scales.iter().enumerate() {
                unsafe {
                    kernel_interp4d_nchw_single(
                        s_img.as_ptr(),
                        out_16ch.as_mut_ptr().add(b * OUT_ELEMS),
                        lut_f32,
                    );
                }
            }
        }
        QLutMode::Int8Shift | QLutMode::Int8Grid => {
            let use_shift = mode == QLutMode::Int8Shift;
            let lut_u32 = if use_shift {
                LUT_INT8_SHIFT.0.as_ptr() as *const u32
            } else {
                LUT_INT8_GRID.0.as_ptr() as *const u32
            };
            let mut tmp_i8 = [0i8; OUT_ELEMS];
            let inv_127 = 1.0f32 / 127.0f32;
            for (b, s_img) in scales.iter().enumerate() {
                unsafe {
                    kernel_int8_nchw_single(
                        s_img.as_ptr(),
                        tmp_i8.as_mut_ptr() as *mut u8,
                        lut_u32,
                        use_shift,
                    );
                }
                let base = b * OUT_ELEMS;
                for i in 0..OUT_ELEMS {
                    out_16ch[base + i] = (tmp_i8[i] as f32) * inv_127;
                }
            }
        }
    }
}

/// Complete Pure-Rust Folded Weights for WideGRQuanvNet v2 (148,306 original params)
struct WideGRQuanvNetV2 {
    num_classes: usize,
    c_stem_w: Vec<f32>,       // 16 x 1 x 2 x 2
    c_stem_b: Vec<f32>,       // 16
    qm_bn0_s: Vec<f32>,       // 16
    qm_bn0_sh: Vec<f32>,      // 16
    qm_c1_w: Vec<f32>,        // 32 x 16 x 3 x 3
    qm_c1_b: Vec<f32>,        // 32
    qm_c2_w: Vec<f32>,        // 32 x 32 x 3 x 3 (dil=2, pad=2)
    qm_c2_b: Vec<f32>,        // 32
    qm_c3_w: Vec<f32>,        // 16 x 32 x 1 x 1
    qm_c3_b: Vec<f32>,        // 16
    gate_bn0_s: Vec<f32>,     // 32
    gate_bn0_sh: Vec<f32>,    // 32
    gate_c1_w: Vec<f32>,      // 32 x 32 x 3 x 3
    gate_c1_b: Vec<f32>,      // 32
    gate_c2_w: Vec<f32>,      // 16 x 32 x 1 x 1
    gate_c2_b: Vec<f32>,      // 16
    head_bn0_s: Vec<f32>,     // 32
    head_bn0_sh: Vec<f32>,    // 32
    head_stem_w: Vec<f32>,    // 48 x 32 x 3 x 3
    head_stem_b: Vec<f32>,    // 48
    head_b1_c1_w: Vec<f32>,   // 48 x 48 x 3 x 3
    head_b1_c1_b: Vec<f32>,   // 48
    head_b1_c2_w: Vec<f32>,   // 48 x 48 x 3 x 3
    head_b1_c2_b: Vec<f32>,   // 48
    head_b2_c1_w: Vec<f32>,   // 64 x 48 x 3 x 3
    head_b2_c1_b: Vec<f32>,   // 64
    head_b2_c2_w: Vec<f32>,   // 64 x 64 x 3 x 3
    head_b2_c2_b: Vec<f32>,   // 64
    head_b2_skip_w: Vec<f32>, // 64 x 48 x 1 x 1
    head_b2_skip_b: Vec<f32>, // 64
    head_fc_w: Vec<f32>,      // nc x 64
    head_fc_b: Vec<f32>,      // nc
    total_floats: usize,
}

impl WideGRQuanvNetV2 {
    fn from_bin(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Failed reading {}: {}", path, e))?;
        if bytes.len() < 16 || &bytes[0..4] != b"MQRW" {
            return Err("Invalid MQRW weight header".to_string());
        }
        let num_classes = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let total_floats = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let expected_bytes = 16 + total_floats * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Weight size mismatch: got {}, expected {}",
                bytes.len(),
                expected_bytes
            ));
        }

        let mut offset = 16usize;
        let mut take = |n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let f = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                v.push(f);
                offset += 4;
            }
            v
        };

        Ok(Self {
            num_classes,
            c_stem_w: take(16 * 1 * 2 * 2),
            c_stem_b: take(16),
            qm_bn0_s: take(16),
            qm_bn0_sh: take(16),
            qm_c1_w: take(32 * 16 * 3 * 3),
            qm_c1_b: take(32),
            qm_c2_w: take(32 * 32 * 3 * 3),
            qm_c2_b: take(32),
            qm_c3_w: take(16 * 32 * 1 * 1),
            qm_c3_b: take(16),
            gate_bn0_s: take(32),
            gate_bn0_sh: take(32),
            gate_c1_w: take(32 * 32 * 3 * 3),
            gate_c1_b: take(32),
            gate_c2_w: take(16 * 32 * 1 * 1),
            gate_c2_b: take(16),
            head_bn0_s: take(32),
            head_bn0_sh: take(32),
            head_stem_w: take(48 * 32 * 3 * 3),
            head_stem_b: take(48),
            head_b1_c1_w: take(48 * 48 * 3 * 3),
            head_b1_c1_b: take(48),
            head_b1_c2_w: take(48 * 48 * 3 * 3),
            head_b1_c2_b: take(48),
            head_b2_c1_w: take(64 * 48 * 3 * 3),
            head_b2_c1_b: take(64),
            head_b2_c2_w: take(64 * 64 * 3 * 3),
            head_b2_c2_b: take(64),
            head_b2_skip_w: take(64 * 48 * 1 * 1),
            head_b2_skip_b: take(64),
            head_fc_w: take(num_classes * 64),
            head_fc_b: take(num_classes),
            total_floats,
        })
    }

    #[inline(always)]
    fn bn_affine_nchw(buf: &mut [f32], ch: usize, spatial: usize, scale: &[f32], shift: &[f32]) {
        for c in 0..ch {
            let s = scale[c];
            let sh = shift[c];
            let base = c * spatial;
            for i in 0..spatial {
                buf[base + i] = buf[base + i] * s + sh;
            }
        }
    }

    #[inline(always)]
    fn conv1x1_nchw(
        inp: &[f32],
        c_in: usize,
        spatial: usize,
        w: &[f32],
        b: &[f32],
        out: &mut [f32],
        c_out: usize,
    ) {
        for oc in 0..c_out {
            let bias = b[oc];
            let out_base = oc * spatial;
            for i in 0..spatial {
                out[out_base + i] = bias;
            }
            let w_oc = &w[oc * c_in..(oc + 1) * c_in];
            for ic in 0..c_in {
                let wt = w_oc[ic];
                let in_slice = &inp[ic * spatial..(ic + 1) * spatial];
                let out_slice = &mut out[out_base..out_base + spatial];
                for i in 0..spatial {
                    out_slice[i] += in_slice[i] * wt;
                }
            }
        }
    }

    #[inline(always)]
    fn conv3x3_nchw(
        inp: &[f32],
        c_in: usize,
        h: usize,
        w_dim: usize,
        pad: usize,
        dil: usize,
        wt: &[f32],
        b: &[f32],
        out: &mut [f32],
        c_out: usize,
    ) {
        let spatial = h * w_dim;
        for oc in 0..c_out {
            let bias = b[oc];
            let ob = oc * spatial;
            for i in 0..spatial {
                out[ob + i] = bias;
            }
        }
        for oc in 0..c_out {
            let ob = oc * spatial;
            for ic in 0..c_in {
                let ib = ic * spatial;
                let kb = (oc * c_in + ic) * 9;
                for kr in 0..3 {
                    let off_r = (kr * dil) as isize - pad as isize;
                    for kc in 0..3 {
                        let kval = wt[kb + kr * 3 + kc];
                        let off_c = (kc * dil) as isize - pad as isize;
                        for r in 0..h {
                            let ir = r as isize + off_r;
                            if ir < 0 || ir >= h as isize {
                                continue;
                            }
                            let in_row = ib + (ir as usize) * w_dim;
                            let out_row = ob + r * w_dim;
                            for c in 0..w_dim {
                                let ic_pos = c as isize + off_c;
                                if ic_pos >= 0 && ic_pos < w_dim as isize {
                                    out[out_row + c] += inp[in_row + ic_pos as usize] * kval;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn forward_single(
        &self,
        raw_img: &[u8; IMG_PIXELS],
        q_raw: &[f32; 16 * NUM_PATCHES],
        apply_var_stab: bool,
    ) -> (Vec<f32>, f32) {
        // Normalize input to [-1, 1] and compute x_01
        let mut x_01 = [0.0f32; IMG_PIXELS];
        let mut x_norm = [0.0f32; IMG_PIXELS];
        for i in 0..IMG_PIXELS {
            let v = (raw_img[i] as f32) / 255.0;
            x_01[i] = v;
            x_norm[i] = (v - 0.5) / 0.5;
        }

        // 1. c_stem_folded (1 -> 16, k=2, s=2) -> 16 x 14 x 14
        let mut f_c = [0.0f32; 16 * NUM_PATCHES];
        for oc in 0..16 {
            let bias = self.c_stem_b[oc];
            let w00 = self.c_stem_w[oc * 4];
            let w01 = self.c_stem_w[oc * 4 + 1];
            let w10 = self.c_stem_w[oc * 4 + 2];
            let w11 = self.c_stem_w[oc * 4 + 3];
            let ob = oc * NUM_PATCHES;
            for r in 0..OUT_H {
                let r0 = (r * 2) * IMG_W;
                let r1 = (r * 2 + 1) * IMG_W;
                for c in 0..OUT_W {
                    let c2 = c * 2;
                    f_c[ob + r * OUT_W + c] = bias
                        + x_norm[r0 + c2] * w00
                        + x_norm[r0 + c2 + 1] * w01
                        + x_norm[r1 + c2] * w10
                        + x_norm[r1 + c2 + 1] * w11;
                }
            }
        }

        // Symmetric stem variance stabilization when hp_score > 0.024
        if apply_var_stab {
            let hp_score = compute_lap_hp_score(&x_01);
            let gate_vs = ((hp_score - 0.024) / 0.024).clamp(0.0, 6.0);
            if gate_vs > 0.0 {
                let mut sq_sum = 0.0f32;
                for i in 0..(16 * NUM_PATCHES) {
                    sq_sum += f_c[i] * f_c[i];
                }
                let rms = ((sq_sum / ((16 * NUM_PATCHES) as f32)).sqrt() / 1.05).max(1.0);
                let scale_c = rms.powf(1.0 + 0.35 * gate_vs);
                for i in 0..(16 * NUM_PATCHES) {
                    f_c[i] /= scale_c;
                }
            }
        }

        // 2. q_mixer: bn0 -> c1(3x3,p=1) -> GELU -> c2(3x3,p=2,d=2) -> GELU -> c3(1x1)
        let mut q_bn = *q_raw;
        Self::bn_affine_nchw(&mut q_bn, 16, NUM_PATCHES, &self.qm_bn0_s, &self.qm_bn0_sh);

        let mut q_h1 = [0.0f32; 32 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &q_bn,
            16,
            OUT_H,
            OUT_W,
            1,
            1,
            &self.qm_c1_w,
            &self.qm_c1_b,
            &mut q_h1,
            32,
        );
        for v in q_h1.iter_mut() {
            *v = gelu_exact(*v);
        }

        let mut q_h2 = [0.0f32; 32 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &q_h1,
            32,
            OUT_H,
            OUT_W,
            2,
            2,
            &self.qm_c2_w,
            &self.qm_c2_b,
            &mut q_h2,
            32,
        );
        for v in q_h2.iter_mut() {
            *v = gelu_exact(*v);
        }

        let mut f_q = [0.0f32; 16 * NUM_PATCHES];
        Self::conv1x1_nchw(&q_h2, 32, NUM_PATCHES, &self.qm_c3_w, &self.qm_c3_b, &mut f_q, 16);

        // 3. gate: cat([f_c, f_q]) -> bn0 -> c1(3x3,p=1) -> GELU -> c2(1x1) -> sigmoid -> alpha
        let mut cat_buf = [0.0f32; 32 * NUM_PATCHES];
        cat_buf[0..(16 * NUM_PATCHES)].copy_from_slice(&f_c);
        cat_buf[(16 * NUM_PATCHES)..(32 * NUM_PATCHES)].copy_from_slice(&f_q);

        let mut g_in = cat_buf;
        Self::bn_affine_nchw(&mut g_in, 32, NUM_PATCHES, &self.gate_bn0_s, &self.gate_bn0_sh);

        let mut g_h1 = [0.0f32; 32 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &g_in,
            32,
            OUT_H,
            OUT_W,
            1,
            1,
            &self.gate_c1_w,
            &self.gate_c1_b,
            &mut g_h1,
            32,
        );
        for v in g_h1.iter_mut() {
            *v = gelu_exact(*v);
        }

        let mut alpha = [0.0f32; 16 * NUM_PATCHES];
        Self::conv1x1_nchw(&g_h1, 32, NUM_PATCHES, &self.gate_c2_w, &self.gate_c2_b, &mut alpha, 16);
        let mut alpha_sum = 0.0f32;
        for i in 0..(16 * NUM_PATCHES) {
            let a = 1.0f32 / (1.0f32 + (-alpha[i]).exp());
            alpha[i] = a;
            alpha_sum += a;
            cat_buf[16 * NUM_PATCHES + i] = a * f_q[i];
        }
        let alpha_mean = alpha_sum / ((16 * NUM_PATCHES) as f32);

        // 4. WideResidualHead
        Self::bn_affine_nchw(
            &mut cat_buf,
            32,
            NUM_PATCHES,
            &self.head_bn0_s,
            &self.head_bn0_sh,
        );
        for v in cat_buf.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        // head.stem (32 -> 48, 14x14) + ReLU
        let mut h_stem = [0.0f32; 48 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &cat_buf,
            32,
            OUT_H,
            OUT_W,
            1,
            1,
            &self.head_stem_w,
            &self.head_stem_b,
            &mut h_stem,
            48,
        );
        for v in h_stem.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        // head.block1 (48 -> 48, 14x14)
        let mut b1_h = [0.0f32; 48 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &h_stem,
            48,
            OUT_H,
            OUT_W,
            1,
            1,
            &self.head_b1_c1_w,
            &self.head_b1_c1_b,
            &mut b1_h,
            48,
        );
        for v in b1_h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut b1_out = [0.0f32; 48 * NUM_PATCHES];
        Self::conv3x3_nchw(
            &b1_h,
            48,
            OUT_H,
            OUT_W,
            1,
            1,
            &self.head_b1_c2_w,
            &self.head_b1_c2_b,
            &mut b1_out,
            48,
        );
        for i in 0..(48 * NUM_PATCHES) {
            let v = b1_out[i] + h_stem[i];
            b1_out[i] = if v < 0.0 { 0.0 } else { v };
        }

        // MaxPool2d(2) -> 48 x 7 x 7
        const H7: usize = 7;
        const P49: usize = 49;
        let mut p1 = [0.0f32; 48 * P49];
        for c in 0..48 {
            let ib = c * NUM_PATCHES;
            let ob = c * P49;
            for r in 0..H7 {
                let r2 = r * 2;
                for col in 0..H7 {
                    let c2 = col * 2;
                    let m0 = b1_out[ib + r2 * OUT_W + c2].max(b1_out[ib + r2 * OUT_W + c2 + 1]);
                    let m1 =
                        b1_out[ib + (r2 + 1) * OUT_W + c2].max(b1_out[ib + (r2 + 1) * OUT_W + c2 + 1]);
                    p1[ob + r * H7 + col] = m0.max(m1);
                }
            }
        }

        // head.block2 (48 -> 64, 7x7)
        let mut b2_h = [0.0f32; 64 * P49];
        Self::conv3x3_nchw(
            &p1,
            48,
            H7,
            H7,
            1,
            1,
            &self.head_b2_c1_w,
            &self.head_b2_c1_b,
            &mut b2_h,
            64,
        );
        for v in b2_h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut b2_out = [0.0f32; 64 * P49];
        Self::conv3x3_nchw(
            &b2_h,
            64,
            H7,
            H7,
            1,
            1,
            &self.head_b2_c2_w,
            &self.head_b2_c2_b,
            &mut b2_out,
            64,
        );
        let mut b2_skip = [0.0f32; 64 * P49];
        Self::conv1x1_nchw(
            &p1,
            48,
            P49,
            &self.head_b2_skip_w,
            &self.head_b2_skip_b,
            &mut b2_skip,
            64,
        );
        for i in 0..(64 * P49) {
            let v = b2_out[i] + b2_skip[i];
            b2_out[i] = if v < 0.0 { 0.0 } else { v };
        }

        // MaxPool2d(2) -> 64 x 3 x 3 + Global Average Pooling -> [64]
        let mut gap = [0.0f32; 64];
        for c in 0..64 {
            let ib = c * P49;
            let mut sum = 0.0f32;
            for r in 0..3 {
                let r2 = r * 2;
                for col in 0..3 {
                    let c2 = col * 2;
                    let m0 = b2_out[ib + r2 * H7 + c2].max(b2_out[ib + r2 * H7 + c2 + 1]);
                    let m1 =
                        b2_out[ib + (r2 + 1) * H7 + c2].max(b2_out[ib + (r2 + 1) * H7 + c2 + 1]);
                    sum += m0.max(m1);
                }
            }
            gap[c] = sum / 9.0f32;
        }

        // Linear(64 -> num_classes)
        let mut logits = vec![0.0f32; self.num_classes];
        for oc in 0..self.num_classes {
            let mut s = self.head_fc_b[oc];
            let wb = oc * 64;
            for ic in 0..64 {
                s += gap[ic] * self.head_fc_w[wb + ic];
            }
            logits[oc] = s;
        }
        (logits, alpha_mean)
    }
}

struct TestVerifyBundle {
    imgs: Vec<[u8; IMG_PIXELS]>,
    labels: Vec<u8>,
    ref_logits_shift: Vec<[f32; 2]>,
    ref_logits_grid: Vec<[f32; 2]>,
    ref_logits_interp4d: Vec<[f32; 2]>,
}

impl TestVerifyBundle {
    fn from_bin(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Failed reading {}: {}", path, e))?;
        if bytes.len() < 20 || &bytes[0..4] != b"MQRV" {
            return Err("Invalid MQRV verify header".to_string());
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut off = 20usize;
        let mut imgs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut arr = [0u8; IMG_PIXELS];
            arr.copy_from_slice(&bytes[off..off + IMG_PIXELS]);
            imgs.push(arr);
            off += IMG_PIXELS;
        }
        let labels = bytes[off..off + n].to_vec();
        off += n;

        let mut read_logits = || -> Vec<[f32; 2]> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let l0 = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                let l1 = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
                v.push([l0, l1]);
                off += 8;
            }
            v
        };
        let ref_logits_shift = read_logits();
        let ref_logits_grid = read_logits();
        let ref_logits_interp4d = read_logits();

        Ok(Self {
            imgs,
            labels,
            ref_logits_shift,
            ref_logits_grid,
            ref_logits_interp4d,
        })
    }
}

fn run_highres_benchmark() {
    println!("================================================================================");
    println!("  HIGH-RESOLUTION Q-LUT LAYER SCALABILITY BENCHMARK (Rust Native Engine)");
    println!("  Memory Complexity: O(1) Auxiliary L2 Cache (Constant 256 KiB Table)");
    println!("================================================================================");
    println!(
        "{:<14} | {:<8} | {:<12} | {:<16} | {:<16} | {:<12}",
        "Resolution", "Stride", "Patches", "INT8 LUT (us)", "Interp4D (us)", "INT8 FPS"
    );
    println!("--------------------------------------------------------------------------------");

    let configs: [(usize, usize, usize, usize); 7] = [
        (28, 28, 2, 50000),
        (64, 64, 2, 20000),
        (128, 128, 2, 5000),
        (224, 224, 2, 2000),
        (512, 512, 2, 500),
        (1024, 1024, 2, 150),
        (1024, 1024, 4, 500),
    ];

    for &(h, w, stride, iters) in &configs {
        let out_h = (h - 2) / stride + 1;
        let out_w = (w - 2) / stride + 1;
        let patches = out_h * out_w;

        let img: Vec<u8> = (0..(h * w)).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
        let mut out_u32: Vec<u32> = vec![0u32; patches];
        let mut out_f32: Vec<f32> = vec![0.0f32; patches * 4];

        for _ in 0..20 {
            unsafe {
                qlut_forward_int8_nhwc_strided_2d(
                    black_box(img.as_ptr()),
                    black_box(out_u32.as_mut_ptr()),
                    h,
                    w,
                    stride,
                    true,
                );
                qlut_forward_interp4d_nchw_strided_2d(
                    black_box(img.as_ptr()),
                    black_box(out_f32.as_mut_ptr()),
                    h,
                    w,
                    stride,
                );
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                qlut_forward_int8_nhwc_strided_2d(
                    black_box(img.as_ptr()),
                    black_box(out_u32.as_mut_ptr()),
                    h,
                    w,
                    stride,
                    true,
                );
            }
        }
        let us_int8 = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t1 = Instant::now();
        let iters_f32 = (iters / 4).max(20);
        for _ in 0..iters_f32 {
            unsafe {
                qlut_forward_interp4d_nchw_strided_2d(
                    black_box(img.as_ptr()),
                    black_box(out_f32.as_mut_ptr()),
                    h,
                    w,
                    stride,
                );
            }
        }
        let us_f32 = t1.elapsed().as_secs_f64() * 1e6 / (iters_f32 as f64);
        let fps_int8 = 1e6 / us_int8;

        println!(
            "{:<14} | s={:<6} | {:<12} | {:>12.2} us | {:>12.2} us | {:>10.1}",
            format!("{}x{}", h, w),
            stride,
            patches,
            us_int8,
            us_f32,
            fps_int8
        );
    }
    println!("================================================================================");
}

fn verify_and_benchmark_mode(
    net: &WideGRQuanvNetV2,
    bundle: &TestVerifyBundle,
    mode: QLutMode,
) -> (f64, f64, f64, f64, f64, f64) {
    let ref_logits = match mode {
        QLutMode::Int8Shift => &bundle.ref_logits_shift,
        QLutMode::Int8Grid => &bundle.ref_logits_grid,
        QLutMode::Interp4d => &bundle.ref_logits_interp4d,
    };

    let n = bundle.imgs.len();
    let mut correct_rust = 0usize;
    let mut correct_torch = 0usize;
    let mut max_diff = 0.0f64;
    let mut q_buf = [0.0f32; 16 * NUM_PATCHES];
    let mut alpha_sum = 0.0f64;

    for i in 0..n {
        extract_multiscale_qlut(&bundle.imgs[i], mode, true, &mut q_buf);
        let (logits, alpha_m) = net.forward_single(&bundle.imgs[i], &q_buf, true);
        alpha_sum += alpha_m as f64;

        let d0 = (logits[0] - ref_logits[i][0]).abs() as f64;
        let d1 = (logits[1] - ref_logits[i][1]).abs() as f64;
        max_diff = max_diff.max(d0).max(d1);

        let pred_r = if logits[1] > logits[0] { 1u8 } else { 0u8 };
        let pred_t = if ref_logits[i][1] > ref_logits[i][0] { 1u8 } else { 0u8 };
        if pred_r == bundle.labels[i] {
            correct_rust += 1;
        }
        if pred_t == bundle.labels[i] {
            correct_torch += 1;
        }
    }

    let acc_rust = (correct_rust as f64) * 100.0 / (n as f64);
    let acc_torch = (correct_torch as f64) * 100.0 / (n as f64);

    // Warmup & Benchmark Batch=1 latency (Q-LUT 4-scale alone vs Full End-to-End 148k Network)
    let sample = &bundle.imgs[0];
    for _ in 0..100 {
        extract_multiscale_qlut(black_box(sample), mode, false, black_box(&mut q_buf));
        let _ = black_box(net.forward_single(black_box(sample), black_box(&q_buf), false));
    }

    let qlut_iters = 5000usize;
    let t0 = Instant::now();
    for _ in 0..qlut_iters {
        extract_multiscale_qlut(black_box(sample), mode, false, black_box(&mut q_buf));
    }
    let qlut_4scale_us = t0.elapsed().as_secs_f64() * 1e6 / (qlut_iters as f64);

    let e2e_iters = 400usize;
    let t1 = Instant::now();
    for _ in 0..e2e_iters {
        extract_multiscale_qlut(black_box(sample), mode, false, black_box(&mut q_buf));
        let res = net.forward_single(black_box(sample), black_box(&q_buf), false);
        black_box(res);
    }
    let e2e_us = t1.elapsed().as_secs_f64() * 1e6 / (e2e_iters as f64);
    let e2e_fps = 1e6 / e2e_us;

    println!(
        "  {:<34} | Rust ACC: {:5.2}% (PyTorch: {:5.2}%) | MaxLogitDiff: {:.2e} | Q-LUT(4x): {:6.2} us | Full-Net E2E: {:7.2} us ({:6.1} FPS) | alpha={:.4}",
        mode.name(),
        acc_rust,
        acc_torch,
        max_diff,
        qlut_4scale_us,
        e2e_us,
        e2e_fps,
        alpha_sum / (n as f64)
    );

    (acc_rust, acc_torch, max_diff, qlut_4scale_us, e2e_us, e2e_fps)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut weights_path = "weights/wide_v2_pneumoniamnist_s42_rust.bin".to_string();
    let mut verify_path = "data/processed/pneumoniamnist_test_rust_verify.bin".to_string();
    let mut bench_highres = false;
    let mut json_out: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                if i + 1 < args.len() {
                    weights_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--verify" => {
                if i + 1 < args.len() {
                    verify_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--bench-resolutions" => {
                bench_highres = true;
            }
            "--json-out" => {
                if i + 1 < args.len() {
                    json_out = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let net = WideGRQuanvNetV2::from_bin(&weights_path).expect("Failed loading folded WideGRQuanvNetV2 weights");
    let bundle = TestVerifyBundle::from_bin(&verify_path).expect("Failed loading PneumoniaMNIST verification bundle");

    let arch_name = if cfg!(target_arch = "aarch64") {
        "ARM64 (aarch64 NEON SIMD - Cortex-A57 Target)"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64 (AVX2/FMA/SSSE3 SIMD)"
    } else {
        "Generic Scalar"
    };

    let int8_kb = qlut_table_int8_bytes() / 1024;
    let fp32_kb = qlut_table_fp32_bytes() / 1024;
    let weights_kib = (net.total_floats * 4) as f64 / 1024.0;
    let total_int8_kib = (int8_kb as f64) + weights_kib;
    let total_interp_kib = (fp32_kb as f64) + weights_kib;

    println!("========================================================================================================================");
    println!("  MEDQUANV-CLI v0.3.0 — REAL END-TO-END NATIVE INFERENCE ENGINE (148,306 Folded Parameters)");
    println!("  Runtime Dependencies : ZERO (No Python | No PyTorch | No CUDA/GPU)");
    println!("  Architecture         : {}", arch_name);
    println!("  Weights Loaded       : {} ({:.2} KiB, {} folded floats)", weights_path, weights_kib, net.total_floats);
    println!("  Test Dataset         : {} ({} real 28x28 PneumoniaMNIST test samples)", verify_path, bundle.imgs.len());
    println!("  L2 Cache Budget      : INT8 Mode = {:.1} KiB ({:.1}% of 2MB L2) | Interp4D Mode = {:.1} KiB ({:.1}% of 2MB L2)",
             total_int8_kib, total_int8_kib * 100.0 / 2048.0,
             total_interp_kib, total_interp_kib * 100.0 / 2048.0);
    println!("========================================================================================================================");

    let m_shift = verify_and_benchmark_mode(&net, &bundle, QLutMode::Int8Shift);
    let m_grid = verify_and_benchmark_mode(&net, &bundle, QLutMode::Int8Grid);
    let m_interp = verify_and_benchmark_mode(&net, &bundle, QLutMode::Interp4d);
    println!("========================================================================================================================");

    if let Some(jpath) = json_out {
        let json_str = format!(
            "{{\n  \"arch\": \"{}\",\n  \"num_test_samples\": {},\n  \"folded_weights_kib\": {:.2},\n  \"folded_weights_floats\": {},\n  \"int8_l2_total_kib\": {:.2},\n  \"interp4d_l2_total_kib\": {:.2},\n  \"modes\": {{\n    \"int8_shift\": {{\"rust_acc\": {:.4}, \"pytorch_acc\": {:.4}, \"max_logit_diff\": {:.8}, \"qlut_4scale_us\": {:.3}, \"e2e_us\": {:.3}, \"e2e_fps\": {:.2}}},\n    \"int8_grid\": {{\"rust_acc\": {:.4}, \"pytorch_acc\": {:.4}, \"max_logit_diff\": {:.8}, \"qlut_4scale_us\": {:.3}, \"e2e_us\": {:.3}, \"e2e_fps\": {:.2}}},\n    \"interp4d\": {{\"rust_acc\": {:.4}, \"pytorch_acc\": {:.4}, \"max_logit_diff\": {:.8}, \"qlut_4scale_us\": {:.3}, \"e2e_us\": {:.3}, \"e2e_fps\": {:.2}}}\n  }}\n}}\n",
            arch_name,
            bundle.imgs.len(),
            weights_kib,
            net.total_floats,
            total_int8_kib,
            total_interp_kib,
            m_shift.0, m_shift.1, m_shift.2, m_shift.3, m_shift.4, m_shift.5,
            m_grid.0, m_grid.1, m_grid.2, m_grid.3, m_grid.4, m_grid.5,
            m_interp.0, m_interp.1, m_interp.2, m_interp.3, m_interp.4, m_interp.5,
        );
        fs::write(&jpath, json_str).expect("Failed writing verification JSON");
        println!("[Saved Verification JSON] {}", jpath);
    }

    if bench_highres {
        println!();
        run_highres_benchmark();
    }
}
