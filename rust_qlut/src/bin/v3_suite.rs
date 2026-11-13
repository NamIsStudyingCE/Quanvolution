//! Pure-Rust Phase B (GR-Quanv v3) & Fair-Play Classical Suite Verification + Latency Benchmark
//! Zero Python | Zero PyTorch | Zero CUDA/GPU Dependencies.

use std::fs;
use std::hint::black_box;
use std::time::Instant;

const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_PIXELS: usize = IMG_H * IMG_W;
const MS_H: usize = 14;
const MS_W: usize = 14;
const MS_HW: usize = MS_H * MS_W; // 196
const LUT_BYTES: usize = 65536 * 4; // 262,144 bytes = 256 KiB

const LAP_K: [[f32; 3]; 3] = [
    [0.25, -0.5, 0.25],
    [-0.5, 1.0, -0.5],
    [0.25, -0.5, 0.25],
];

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn u32(&mut self) -> u32 {
        let b = &self.buf[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn vec_f32(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let b = &self.buf[self.pos..self.pos + 4];
            self.pos += 4;
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        out
    }
}

#[derive(Copy, Clone)]
enum Act {
    None,
    Relu,
    Gelu,
}

#[inline(always)]
fn gelu_exact(x: f32) -> f32 {
    let ax = (x as f64).abs() * 0.70710678118654752440;
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592;
    let erf_abs = 1.0 - poly * t * (-ax * ax).exp();
    let erf_val = if x < 0.0 { -erf_abs } else { erf_abs };
    (0.5 * (x as f64) * (1.0 + erf_val)) as f32
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn bn_affine(buf: &mut [f32], s: &[f32], sh: &[f32], hw: usize, relu_after: bool) {
    let ch = s.len();
    for c in 0..ch {
        let sc = s[c];
        let sf = sh[c];
        let row = &mut buf[c * hw..(c + 1) * hw];
        for v in row.iter_mut() {
            let y = (*v) * sc + sf;
            *v = if relu_after && y < 0.0 { 0.0 } else { y };
        }
    }
}

#[inline]
fn bn_affine_into(inp: &[f32], out: &mut [f32], s: &[f32], sh: &[f32], hw: usize, relu_after: bool) {
    let ch = s.len();
    for c in 0..ch {
        let sc = s[c];
        let sf = sh[c];
        let src = &inp[c * hw..(c + 1) * hw];
        let dst = &mut out[c * hw..(c + 1) * hw];
        for p in 0..hw {
            let y = src[p] * sc + sf;
            dst[p] = if relu_after && y < 0.0 { 0.0 } else { y };
        }
    }
}

#[inline]
fn conv1x1(
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    c_in: usize,
    c_out: usize,
    hw: usize,
    relu: bool,
) {
    for oc in 0..c_out {
        let o_row = &mut out[oc * hw..(oc + 1) * hw];
        o_row.fill(b[oc]);
        let w_row = &w[oc * c_in..(oc + 1) * c_in];
        for ic in 0..c_in {
            let wt = w_row[ic];
            let i_row = &inp[ic * hw..(ic + 1) * hw];
            for p in 0..hw {
                o_row[p] += wt * i_row[p];
            }
        }
        if relu {
            for v in o_row.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

#[inline]
fn conv_dw3x3(
    inp: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    out: &mut [f32],
    c: usize,
    h: usize,
    w_dim: usize,
    dilation: usize,
    pad: usize,
    relu: bool,
) {
    let hw = h * w_dim;
    let hi = h as isize;
    let wi = w_dim as isize;
    let d_is = dilation as isize;
    let p_is = pad as isize;

    for ch in 0..c {
        let bias = match b {
            Some(bv) => bv[ch],
            None => 0.0,
        };
        let k = &w[ch * 9..(ch + 1) * 9];
        let src = &inp[ch * hw..(ch + 1) * hw];
        let dst = &mut out[ch * hw..(ch + 1) * hw];
        for r in 0..h {
            for col in 0..w_dim {
                let mut acc = bias;
                for kr in 0..3isize {
                    let ir = r as isize - p_is + kr * d_is;
                    if ir >= 0 && ir < hi {
                        let row_off = (ir as usize) * w_dim;
                        let k_off = (kr as usize) * 3;
                        for kc in 0..3isize {
                            let ic = col as isize - p_is + kc * d_is;
                            if ic >= 0 && ic < wi {
                                acc += src[row_off + ic as usize] * k[k_off + kc as usize];
                            }
                        }
                    }
                }
                dst[r * w_dim + col] = if relu && acc < 0.0 { 0.0 } else { acc };
            }
        }
    }
}

#[inline]
fn conv_grouped3x3(
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w_dim: usize,
) {
    let hw = h * w_dim;
    let mult = c_out / c_in;
    for oc in 0..c_out {
        let ic_g = oc / mult;
        let bias = b[oc];
        let k = &w[oc * 9..(oc + 1) * 9];
        let src = &inp[ic_g * hw..(ic_g + 1) * hw];
        let dst = &mut out[oc * hw..(oc + 1) * hw];
        for r in 0..h {
            for col in 0..w_dim {
                let mut acc = bias;
                for kr in 0..3isize {
                    let ir = r as isize - 1 + kr;
                    if ir >= 0 && ir < h as isize {
                        let row_off = (ir as usize) * w_dim;
                        let k_off = (kr as usize) * 3;
                        for kc in 0..3isize {
                            let ic = col as isize - 1 + kc;
                            if ic >= 0 && ic < w_dim as isize {
                                acc += src[row_off + ic as usize] * k[k_off + kc as usize];
                            }
                        }
                    }
                }
                dst[r * w_dim + col] = acc;
            }
        }
    }
}

#[inline]
fn conv3x3(
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w_dim: usize,
    dilation: usize,
    pad: usize,
    act: Act,
) {
    let hw = h * w_dim;
    let hi = h as isize;
    let wi = w_dim as isize;
    let d_is = dilation as isize;
    let p_is = pad as isize;

    for oc in 0..c_out {
        let dst = &mut out[oc * hw..(oc + 1) * hw];
        dst.fill(b[oc]);
        for ic_ch in 0..c_in {
            let src = &inp[ic_ch * hw..(ic_ch + 1) * hw];
            let k = &w[(oc * c_in + ic_ch) * 9..(oc * c_in + ic_ch + 1) * 9];
            for kr in 0..3isize {
                let k_off = (kr as usize) * 3;
                for r in 0..h {
                    let ir = r as isize - p_is + kr * d_is;
                    if ir >= 0 && ir < hi {
                        let src_row = &src[(ir as usize) * w_dim..(ir as usize + 1) * w_dim];
                        let dst_row = &mut dst[r * w_dim..(r + 1) * w_dim];
                        for kc in 0..3isize {
                            let wt = k[k_off + kc as usize];
                            let c_shift = kc * d_is - p_is;
                            for col in 0..w_dim {
                                let ic = col as isize + c_shift;
                                if ic >= 0 && ic < wi {
                                    dst_row[col] += src_row[ic as usize] * wt;
                                }
                            }
                        }
                    }
                }
            }
        }
        match act {
            Act::None => {}
            Act::Relu => {
                for v in dst.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Act::Gelu => {
                for v in dst.iter_mut() {
                    *v = gelu_exact(*v);
                }
            }
        }
    }
}

#[inline]
fn conv_c_stem(x_norm: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    for oc in 0..16 {
        let dst = &mut out[oc * MS_HW..(oc + 1) * MS_HW];
        let k = &w[oc * 4..oc * 4 + 4];
        let bias = b[oc];
        for r in 0..MS_H {
            let r0 = (2 * r) * IMG_W;
            let r1 = (2 * r + 1) * IMG_W;
            for c in 0..MS_W {
                let c0 = 2 * c;
                dst[r * MS_W + c] = bias
                    + k[0] * x_norm[r0 + c0]
                    + k[1] * x_norm[r0 + c0 + 1]
                    + k[2] * x_norm[r1 + c0]
                    + k[3] * x_norm[r1 + c0 + 1];
            }
        }
    }
}

#[inline]
fn conv_classical_branches(
    x_norm: &[f32],
    br_w: &[Vec<f32>; 4],
    br_b: &[Vec<f32>; 4],
    out: &mut [f32],
) {
    // Scale 0: k=2, s=2, p=0, d=1
    for oc in 0..4 {
        let dst = &mut out[oc * MS_HW..(oc + 1) * MS_HW];
        let k = &br_w[0][oc * 4..oc * 4 + 4];
        let bias = br_b[0][oc];
        for r in 0..MS_H {
            let r0 = (2 * r) * IMG_W;
            let r1 = (2 * r + 1) * IMG_W;
            for c in 0..MS_W {
                let c0 = 2 * c;
                dst[r * MS_W + c] = bias
                    + k[0] * x_norm[r0 + c0]
                    + k[1] * x_norm[r0 + c0 + 1]
                    + k[2] * x_norm[r1 + c0]
                    + k[3] * x_norm[r1 + c0 + 1];
            }
        }
    }
    // Scale 1: k=2, s=2, p=1, d=2
    for oc in 0..4 {
        let ch = 4 + oc;
        let dst = &mut out[ch * MS_HW..(ch + 1) * MS_HW];
        let k = &br_w[1][oc * 4..oc * 4 + 4];
        let bias = br_b[1][oc];
        for r in 0..MS_H {
            for c in 0..MS_W {
                let mut acc = bias;
                for kr in 0..2isize {
                    let ir = 2 * (r as isize) - 1 + 2 * kr;
                    if ir >= 0 && ir < IMG_H as isize {
                        let r_off = (ir as usize) * IMG_W;
                        for kc in 0..2isize {
                            let ic = 2 * (c as isize) - 1 + 2 * kc;
                            if ic >= 0 && ic < IMG_W as isize {
                                acc += x_norm[r_off + ic as usize] * k[(kr * 2 + kc) as usize];
                            }
                        }
                    }
                }
                dst[r * MS_W + c] = acc;
            }
        }
    }
    // Scale 2: k=2, s=2, p=2, d=3 -> sliced :14, :14
    for oc in 0..4 {
        let ch = 8 + oc;
        let dst = &mut out[ch * MS_HW..(ch + 1) * MS_HW];
        let k = &br_w[2][oc * 4..oc * 4 + 4];
        let bias = br_b[2][oc];
        for r in 0..MS_H {
            for c in 0..MS_W {
                let mut acc = bias;
                for kr in 0..2isize {
                    let ir = 2 * (r as isize) - 2 + 3 * kr;
                    if ir >= 0 && ir < IMG_H as isize {
                        let r_off = (ir as usize) * IMG_W;
                        for kc in 0..2isize {
                            let ic = 2 * (c as isize) - 2 + 3 * kc;
                            if ic >= 0 && ic < IMG_W as isize {
                                acc += x_norm[r_off + ic as usize] * k[(kr * 2 + kc) as usize];
                            }
                        }
                    }
                }
                dst[r * MS_W + c] = acc;
            }
        }
    }
    // Scale 3: k=3, s=2, p=1, d=1
    for oc in 0..4 {
        let ch = 12 + oc;
        let dst = &mut out[ch * MS_HW..(ch + 1) * MS_HW];
        let k = &br_w[3][oc * 9..oc * 9 + 9];
        let bias = br_b[3][oc];
        for r in 0..MS_H {
            for c in 0..MS_W {
                let mut acc = bias;
                for kr in 0..3isize {
                    let ir = 2 * (r as isize) - 1 + kr;
                    if ir >= 0 && ir < IMG_H as isize {
                        let r_off = (ir as usize) * IMG_W;
                        for kc in 0..3isize {
                            let ic = 2 * (c as isize) - 1 + kc;
                            if ic >= 0 && ic < IMG_W as isize {
                                acc += x_norm[r_off + ic as usize] * k[(kr * 3 + kc) as usize];
                            }
                        }
                    }
                }
                dst[r * MS_W + c] = acc;
            }
        }
    }
}

#[inline]
fn compute_hp_score_x01(x_01: &[f32]) -> f32 {
    let reflect = |idx: isize, len: isize| -> usize {
        if idx < 0 {
            (-idx) as usize
        } else if idx >= len {
            (2 * len - 2 - idx) as usize
        } else {
            idx as usize
        }
    };
    let mut sum_abs = 0.0f32;
    for r in 0..IMG_H as isize {
        for c in 0..IMG_W as isize {
            let mut acc = 0.0f32;
            for kr in 0..3isize {
                let ir = reflect(r + kr - 1, IMG_H as isize);
                let row_off = ir * IMG_W;
                for kc in 0..3isize {
                    let ic = reflect(c + kc - 1, IMG_W as isize);
                    acc += x_01[row_off + ic] * LAP_K[kr as usize][kc as usize];
                }
            }
            sum_abs += acc.abs();
        }
    }
    sum_abs / (IMG_PIXELS as f32)
}

#[inline]
fn stabilize_f_c_and_cdrop(x_01: &[f32], f_c: &mut [f32], with_cdrop: bool) -> f32 {
    let hp_score = compute_hp_score_x01(x_01);
    let tau = 0.024f32;
    let excess = ((hp_score - tau) / tau).clamp(0.0, 6.0);
    if excess <= 0.0 {
        return 1.0;
    }
    let mut sum_sq = 0.0f32;
    for v in f_c.iter() {
        sum_sq += (*v) * (*v);
    }
    let mean_sq = sum_sq / (f_c.len() as f32);
    let rms_c = (mean_sq.sqrt() / 1.05).max(1.0);
    let scale_c = rms_c.powf(1.0 + 0.35 * excess);
    let inv_s = 1.0 / scale_c;
    for v in f_c.iter_mut() {
        *v *= inv_s;
    }
    if with_cdrop {
        let soft = excess * ((excess - 0.15) / 0.50).clamp(0.0, 1.0);
        (1.0 / (1.0 + 0.25 * soft)).max(0.40)
    } else {
        1.0
    }
}

#[inline]
fn maxpool2x2(inp: &[f32], out: &mut [f32], c: usize, h: usize, w_dim: usize) {
    let oh = h / 2;
    let ow = w_dim / 2;
    let ihw = h * w_dim;
    let ohw = oh * ow;
    for ch in 0..c {
        let src = &inp[ch * ihw..(ch + 1) * ihw];
        let dst = &mut out[ch * ohw..(ch + 1) * ohw];
        for r in 0..oh {
            let r0 = (2 * r) * w_dim;
            let r1 = (2 * r + 1) * w_dim;
            for col in 0..ow {
                let c0 = 2 * col;
                let a = src[r0 + c0];
                let b = src[r0 + c0 + 1];
                let d = src[r1 + c0];
                let e = src[r1 + c0 + 1];
                dst[r * ow + col] = a.max(b).max(d).max(e);
            }
        }
    }
}

#[inline]
fn avgpool2x2(inp: &[f32], out: &mut [f32], c: usize, h: usize, w_dim: usize) {
    let oh = h / 2;
    let ow = w_dim / 2;
    let ihw = h * w_dim;
    let ohw = oh * ow;
    for ch in 0..c {
        let src = &inp[ch * ihw..(ch + 1) * ihw];
        let dst = &mut out[ch * ohw..(ch + 1) * ohw];
        for r in 0..oh {
            let r0 = (2 * r) * w_dim;
            let r1 = (2 * r + 1) * w_dim;
            for col in 0..ow {
                let c0 = 2 * col;
                dst[r * ow + col] =
                    0.25 * (src[r0 + c0] + src[r0 + c0 + 1] + src[r1 + c0] + src[r1 + c0 + 1]);
            }
        }
    }
}

#[inline]
fn gap_and_fc(inp: &[f32], w: &[f32], b: &[f32], c: usize, hw: usize, nc: usize, gap_buf: &mut [f32]) -> [f32; 8] {
    let inv_hw = 1.0 / (hw as f32);
    for ch in 0..c {
        let src = &inp[ch * hw..(ch + 1) * hw];
        let mut s = 0.0f32;
        for v in src.iter() {
            s += *v;
        }
        gap_buf[ch] = s * inv_hw;
    }
    let mut logits = [0.0f32; 8];
    for oc in 0..nc {
        let mut acc = b[oc];
        let wr = &w[oc * c..(oc + 1) * c];
        for ic in 0..c {
            acc += wr[ic] * gap_buf[ic];
        }
        logits[oc] = acc;
    }
    logits
}

#[inline]
fn qlut_stage1_extract(x_01: &[f32], lut: &[i8], q1: &mut [f32], bins: &mut [usize; IMG_PIXELS]) {
    for idx in 0..IMG_PIXELS {
        bins[idx] = (((x_01[idx] * 255.0 + 1e-4).floor().clamp(0.0, 255.0)) as usize) >> 4;
    }
    const SCALE: f32 = 1.0 / 127.0;
    for r in 0..MS_H {
        let ri = r as isize;
        let r_d1_0 = 2 * r;
        let r_d1_1 = 2 * r + 1;
        let r_d2_0 = (2 * ri - 1).clamp(0, 27) as usize;
        let r_d2_1 = (2 * ri + 1).clamp(0, 27) as usize;
        let r_d3_0 = (2 * ri - 2).clamp(0, 27) as usize;
        let r_ax_0 = r_d2_0;
        let r_ax_1 = 2 * r;
        let r_ax_2 = r_d2_1;

        for c in 0..MS_W {
            let ci = c as isize;
            let p_idx = r * MS_W + c;

            // Scale 0: d=1
            let i0 = (bins[r_d1_0 * IMG_W + 2 * c] << 12)
                | (bins[r_d1_0 * IMG_W + 2 * c + 1] << 8)
                | (bins[r_d1_1 * IMG_W + 2 * c] << 4)
                | bins[r_d1_1 * IMG_W + 2 * c + 1];
            let l0 = i0 * 4;
            q1[0 * MS_HW + p_idx] = (lut[l0] as f32) * SCALE;
            q1[1 * MS_HW + p_idx] = (lut[l0 + 1] as f32) * SCALE;
            q1[2 * MS_HW + p_idx] = (lut[l0 + 2] as f32) * SCALE;
            q1[3 * MS_HW + p_idx] = (lut[l0 + 3] as f32) * SCALE;

            // Scale 1: d=2
            let c_d2_0 = (2 * ci - 1).clamp(0, 27) as usize;
            let c_d2_1 = (2 * ci + 1).clamp(0, 27) as usize;
            let i1 = (bins[r_d2_0 * IMG_W + c_d2_0] << 12)
                | (bins[r_d2_0 * IMG_W + c_d2_1] << 8)
                | (bins[r_d2_1 * IMG_W + c_d2_0] << 4)
                | bins[r_d2_1 * IMG_W + c_d2_1];
            let l1 = i1 * 4;
            q1[4 * MS_HW + p_idx] = (lut[l1] as f32) * SCALE;
            q1[5 * MS_HW + p_idx] = (lut[l1 + 1] as f32) * SCALE;
            q1[6 * MS_HW + p_idx] = (lut[l1 + 2] as f32) * SCALE;
            q1[7 * MS_HW + p_idx] = (lut[l1 + 3] as f32) * SCALE;

            // Scale 2: d=3
            let c_d3_0 = (2 * ci - 2).clamp(0, 27) as usize;
            let i2 = (bins[r_d3_0 * IMG_W + c_d3_0] << 12)
                | (bins[r_d3_0 * IMG_W + c_d2_1] << 8)
                | (bins[r_d2_1 * IMG_W + c_d3_0] << 4)
                | bins[r_d2_1 * IMG_W + c_d2_1];
            let l2 = i2 * 4;
            q1[8 * MS_HW + p_idx] = (lut[l2] as f32) * SCALE;
            q1[9 * MS_HW + p_idx] = (lut[l2 + 1] as f32) * SCALE;
            q1[10 * MS_HW + p_idx] = (lut[l2 + 2] as f32) * SCALE;
            q1[11 * MS_HW + p_idx] = (lut[l2 + 3] as f32) * SCALE;

            // Scale 3: axial
            let c_ax_1 = 2 * c;
            let c_ax_2 = (2 * ci + 1).clamp(0, 27) as usize;
            let i3 = (bins[r_ax_0 * IMG_W + c_ax_1] << 12)
                | (bins[r_ax_1 * IMG_W + c_ax_1] << 8)
                | (bins[r_ax_2 * IMG_W + c_ax_1] << 4)
                | bins[r_ax_1 * IMG_W + c_ax_2];
            let l3 = i3 * 4;
            q1[12 * MS_HW + p_idx] = (lut[l3] as f32) * SCALE;
            q1[13 * MS_HW + p_idx] = (lut[l3 + 1] as f32) * SCALE;
            q1[14 * MS_HW + p_idx] = (lut[l3 + 2] as f32) * SCALE;
            q1[15 * MS_HW + p_idx] = (lut[l3 + 3] as f32) * SCALE;
        }
    }
}

#[inline]
fn qlut_stage2_extract(s_map: &[f32], lut: &[i8], q2: &mut [f32], bins: &mut [usize; 4 * MS_HW]) {
    for i in 0..(4 * MS_HW) {
        bins[i] = (((sigmoid(s_map[i]) * 255.0 + 1e-4).floor().clamp(0.0, 255.0)) as usize) >> 4;
    }
    const SCALE: f32 = 1.0 / 127.0;
    for ch in 0..4 {
        let cb = &bins[ch * MS_HW..(ch + 1) * MS_HW];
        let o_base = ch * 4;
        for r in 0..MS_H {
            let r1 = (r + 1).min(MS_H - 1);
            for c in 0..MS_W {
                let c1 = (c + 1).min(MS_W - 1);
                let l = ((cb[r * MS_W + c] << 12)
                    | (cb[r * MS_W + c1] << 8)
                    | (cb[r1 * MS_W + c] << 4)
                    | cb[r1 * MS_W + c1])
                    * 4;
                let p_idx = r * MS_W + c;
                q2[(o_base + 0) * MS_HW + p_idx] = (lut[l] as f32) * SCALE;
                q2[(o_base + 1) * MS_HW + p_idx] = (lut[l + 1] as f32) * SCALE;
                q2[(o_base + 2) * MS_HW + p_idx] = (lut[l + 2] as f32) * SCALE;
                q2[(o_base + 3) * MS_HW + p_idx] = (lut[l + 3] as f32) * SCALE;
            }
        }
    }
}

fn load_lut_i8(path: &str) -> Vec<i8> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("Failed reading LUT {}: {}", path, e));
    assert_eq!(bytes.len(), LUT_BYTES, "Unexpected LUT size for {}", path);
    bytes.into_iter().map(|b| b as i8).collect()
}

struct CommonV3Head {
    c_stem_w: Vec<f32>,
    c_stem_b: Vec<f32>,
    q_in_s: Vec<f32>,
    q_in_sh: Vec<f32>,
    qm_dw1_w: Vec<f32>,
    qm_pw1_w: Vec<f32>,
    qm_pw1_b: Vec<f32>,
    qm_dw2_w: Vec<f32>,
    qm_pw2_w: Vec<f32>,
    qm_pw2_b: Vec<f32>,
    gate_bn_s: Vec<f32>,
    gate_bn_sh: Vec<f32>,
    gate_dw_w: Vec<f32>,
    gate_pw_w: Vec<f32>,
    gate_pw_b: Vec<f32>,
    // Stage-2 branch depends on kind:
    s2_w0: Vec<f32>,
    s2_b0: Vec<f32>,
    s2_w1: Vec<f32>, // norm_q2_s OR s2_gdw_w
    s2_b1: Vec<f32>, // norm_q2_sh OR s2_gdw_b
    head_stem_w: Vec<f32>,
    head_stem_b: Vec<f32>,
    b1_dw1_w: Vec<f32>,
    b1_pw1_w: Vec<f32>,
    b1_pw1_b: Vec<f32>,
    b1_dw2_w: Vec<f32>,
    b1_pw2_w: Vec<f32>,
    b1_pw2_b: Vec<f32>,
    b2_dw1_w: Vec<f32>,
    b2_pw1_w: Vec<f32>,
    b2_pw1_b: Vec<f32>,
    b2_dw2_w: Vec<f32>,
    b2_pw2_w: Vec<f32>,
    b2_pw2_b: Vec<f32>,
    b2_skip_w: Vec<f32>,
    b2_skip_b: Vec<f32>,
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

struct QuanvV3Net {
    nc: usize,
    weight_bytes: usize,
    lut: Vec<i8>,
    h: CommonV3Head,
}

struct CtrlV3LightNet {
    nc: usize,
    weight_bytes: usize,
    br_w: [Vec<f32>; 4],
    br_b: [Vec<f32>; 4],
    h: CommonV3Head,
}

fn read_tail_common(r: &mut Reader, nc: usize, w1: usize, w2: usize) -> (
    Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
    Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
    Vec<f32>, Vec<f32>,
) {
    (
        r.vec_f32(w1 * 48), r.vec_f32(w1),
        r.vec_f32(w1 * 9), r.vec_f32(w1 * w1), r.vec_f32(w1),
        r.vec_f32(w1 * 9), r.vec_f32(w1 * w1), r.vec_f32(w1),
        r.vec_f32(w1 * 9), r.vec_f32(w2 * w1), r.vec_f32(w2),
        r.vec_f32(w2 * 9), r.vec_f32(w2 * w2), r.vec_f32(w2),
        r.vec_f32(w2 * w1), r.vec_f32(w2),
        r.vec_f32(nc * w2), r.vec_f32(nc),
    )
}

impl QuanvV3Net {
    fn from_bin(w_path: &str, lut_path: &str) -> Self {
        let bytes = fs::read(w_path).expect("read quanv_v3 weights");
        assert_eq!(&bytes[0..4], b"MQV3");
        let mut r = Reader::new(&bytes[4..]);
        let (_ver, kid, nc, _tot) = (r.u32(), r.u32(), r.u32() as usize, r.u32());
        assert_eq!(kid, 1);
        let c_stem_w = r.vec_f32(64); let c_stem_b = r.vec_f32(16);
        let q_in_s = r.vec_f32(16); let q_in_sh = r.vec_f32(16);
        let qm_dw1_w = r.vec_f32(144);
        let qm_pw1_w = r.vec_f32(512); let qm_pw1_b = r.vec_f32(32);
        let qm_dw2_w = r.vec_f32(288);
        let qm_pw2_w = r.vec_f32(512); let qm_pw2_b = r.vec_f32(16);
        let gate_bn_s = r.vec_f32(32); let gate_bn_sh = r.vec_f32(32);
        let gate_dw_w = r.vec_f32(288);
        let gate_pw_w = r.vec_f32(512); let gate_pw_b = r.vec_f32(16);
        let s2_w0 = r.vec_f32(64); let s2_b0 = r.vec_f32(4);
        let s2_w1 = r.vec_f32(16); let s2_b1 = r.vec_f32(16);
        let (head_stem_w, head_stem_b, b1_dw1_w, b1_pw1_w, b1_pw1_b, b1_dw2_w, b1_pw2_w, b1_pw2_b,
             b2_dw1_w, b2_pw1_w, b2_pw1_b, b2_dw2_w, b2_pw2_w, b2_pw2_b, b2_skip_w, b2_skip_b, fc_w, fc_b) =
            read_tail_common(&mut r, nc, 64, 96);
        Self {
            nc,
            weight_bytes: bytes.len() - 20,
            lut: load_lut_i8(lut_path),
            h: CommonV3Head {
                c_stem_w, c_stem_b, q_in_s, q_in_sh, qm_dw1_w, qm_pw1_w, qm_pw1_b, qm_dw2_w, qm_pw2_w, qm_pw2_b,
                gate_bn_s, gate_bn_sh, gate_dw_w, gate_pw_w, gate_pw_b, s2_w0, s2_b0, s2_w1, s2_b1,
                head_stem_w, head_stem_b, b1_dw1_w, b1_pw1_w, b1_pw1_b, b1_dw2_w, b1_pw2_w, b1_pw2_b,
                b2_dw1_w, b2_pw1_w, b2_pw1_b, b2_dw2_w, b2_pw2_w, b2_pw2_b, b2_skip_w, b2_skip_b, fc_w, fc_b,
            },
        }
    }
}

impl CtrlV3LightNet {
    fn from_bin(w_path: &str) -> Self {
        let bytes = fs::read(w_path).expect("read ctrl_v3 weights");
        assert_eq!(&bytes[0..4], b"MQV3");
        let mut r = Reader::new(&bytes[4..]);
        let (_ver, kid, nc, _tot) = (r.u32(), r.u32(), r.u32() as usize, r.u32());
        assert_eq!(kid, 2);
        let c_stem_w = r.vec_f32(64); let c_stem_b = r.vec_f32(16);
        let br0_w = r.vec_f32(16); let br0_b = r.vec_f32(4);
        let br1_w = r.vec_f32(16); let br1_b = r.vec_f32(4);
        let br2_w = r.vec_f32(16); let br2_b = r.vec_f32(4);
        let br3_w = r.vec_f32(36); let br3_b = r.vec_f32(4);
        let q_in_s = r.vec_f32(16); let q_in_sh = r.vec_f32(16);
        let qm_dw1_w = r.vec_f32(144);
        let qm_pw1_w = r.vec_f32(512); let qm_pw1_b = r.vec_f32(32);
        let qm_dw2_w = r.vec_f32(288);
        let qm_pw2_w = r.vec_f32(512); let qm_pw2_b = r.vec_f32(16);
        let gate_bn_s = r.vec_f32(32); let gate_bn_sh = r.vec_f32(32);
        let gate_dw_w = r.vec_f32(288);
        let gate_pw_w = r.vec_f32(512); let gate_pw_b = r.vec_f32(16);
        let s2_w0 = r.vec_f32(64); let s2_b0 = r.vec_f32(4);
        let s2_w1 = r.vec_f32(144); let s2_b1 = r.vec_f32(16);
        let (head_stem_w, head_stem_b, b1_dw1_w, b1_pw1_w, b1_pw1_b, b1_dw2_w, b1_pw2_w, b1_pw2_b,
             b2_dw1_w, b2_pw1_w, b2_pw1_b, b2_dw2_w, b2_pw2_w, b2_pw2_b, b2_skip_w, b2_skip_b, fc_w, fc_b) =
            read_tail_common(&mut r, nc, 64, 96);
        Self {
            nc,
            weight_bytes: bytes.len() - 20,
            br_w: [br0_w, br1_w, br2_w, br3_w],
            br_b: [br0_b, br1_b, br2_b, br3_b],
            h: CommonV3Head {
                c_stem_w, c_stem_b, q_in_s, q_in_sh, qm_dw1_w, qm_pw1_w, qm_pw1_b, qm_dw2_w, qm_pw2_w, qm_pw2_b,
                gate_bn_s, gate_bn_sh, gate_dw_w, gate_pw_w, gate_pw_b, s2_w0, s2_b0, s2_w1, s2_b1,
                head_stem_w, head_stem_b, b1_dw1_w, b1_pw1_w, b1_pw1_b, b1_dw2_w, b1_pw2_w, b1_pw2_b,
                b2_dw1_w, b2_pw1_w, b2_pw1_b, b2_dw2_w, b2_pw2_w, b2_pw2_b, b2_skip_w, b2_skip_b, fc_w, fc_b,
            },
        }
    }
}

struct CtrlV2FullNet {
    nc: usize,
    weight_bytes: usize,
    c_stem_w: Vec<f32>, c_stem_b: Vec<f32>,
    br_w: [Vec<f32>; 4], br_b: [Vec<f32>; 4],
    mx_bn0_s: Vec<f32>, mx_bn0_sh: Vec<f32>,
    mx_c1_w: Vec<f32>, mx_c1_b: Vec<f32>,
    mx_c2_w: Vec<f32>, mx_c2_b: Vec<f32>,
    mx_c3_w: Vec<f32>, mx_c3_b: Vec<f32>,
    gate_bn_s: Vec<f32>, gate_bn_sh: Vec<f32>,
    gate_c1_w: Vec<f32>, gate_c1_b: Vec<f32>,
    gate_c2_w: Vec<f32>, gate_c2_b: Vec<f32>,
    head_pre_s: Vec<f32>, head_pre_sh: Vec<f32>,
    head_stem_w: Vec<f32>, head_stem_b: Vec<f32>,
    b1_c1_w: Vec<f32>, b1_c1_b: Vec<f32>,
    b1_c2_w: Vec<f32>, b1_c2_b: Vec<f32>,
    b2_c1_w: Vec<f32>, b2_c1_b: Vec<f32>,
    b2_c2_w: Vec<f32>, b2_c2_b: Vec<f32>,
    b2_skip_w: Vec<f32>, b2_skip_b: Vec<f32>,
    fc_w: Vec<f32>, fc_b: Vec<f32>,
}

impl CtrlV2FullNet {
    fn from_bin(w_path: &str) -> Self {
        let bytes = fs::read(w_path).expect("read ctrl_v2 weights");
        assert_eq!(&bytes[0..4], b"MQV3");
        let mut r = Reader::new(&bytes[4..]);
        let (_ver, kid, nc, _tot) = (r.u32(), r.u32(), r.u32() as usize, r.u32());
        assert_eq!(kid, 3);
        let c_stem_w = r.vec_f32(64);
        let c_stem_b = r.vec_f32(16);
        let (br0_w, br0_b) = (r.vec_f32(16), r.vec_f32(4));
        let (br1_w, br1_b) = (r.vec_f32(16), r.vec_f32(4));
        let (br2_w, br2_b) = (r.vec_f32(16), r.vec_f32(4));
        let (br3_w, br3_b) = (r.vec_f32(36), r.vec_f32(4));
        Self {
            nc,
            weight_bytes: bytes.len() - 20,
            c_stem_w, c_stem_b,
            br_w: [br0_w, br1_w, br2_w, br3_w],
            br_b: [br0_b, br1_b, br2_b, br3_b],
            mx_bn0_s: r.vec_f32(16), mx_bn0_sh: r.vec_f32(16),
            mx_c1_w: r.vec_f32(32 * 16 * 9), mx_c1_b: r.vec_f32(32),
            mx_c2_w: r.vec_f32(32 * 32 * 9), mx_c2_b: r.vec_f32(32),
            mx_c3_w: r.vec_f32(16 * 32), mx_c3_b: r.vec_f32(16),
            gate_bn_s: r.vec_f32(32), gate_bn_sh: r.vec_f32(32),
            gate_c1_w: r.vec_f32(32 * 32 * 9), gate_c1_b: r.vec_f32(32),
            gate_c2_w: r.vec_f32(16 * 32), gate_c2_b: r.vec_f32(16),
            head_pre_s: r.vec_f32(32), head_pre_sh: r.vec_f32(32),
            head_stem_w: r.vec_f32(48 * 32 * 9), head_stem_b: r.vec_f32(48),
            b1_c1_w: r.vec_f32(48 * 48 * 9), b1_c1_b: r.vec_f32(48),
            b1_c2_w: r.vec_f32(48 * 48 * 9), b1_c2_b: r.vec_f32(48),
            b2_c1_w: r.vec_f32(64 * 48 * 9), b2_c1_b: r.vec_f32(64),
            b2_c2_w: r.vec_f32(64 * 64 * 9), b2_c2_b: r.vec_f32(64),
            b2_skip_w: r.vec_f32(64 * 48), b2_skip_b: r.vec_f32(64),
            fc_w: r.vec_f32(nc * 64), fc_b: r.vec_f32(nc),
        }
    }
}

struct MicroNet {
    nc: usize,
    weight_bytes: usize,
    lut: Option<Vec<i8>>,
    br_w: Option<[Vec<f32>; 4]>,
    br_b: Option<[Vec<f32>; 4]>,
    stem0_s: Vec<f32>, stem0_sh: Vec<f32>,
    pw1_w: Vec<f32>, pw1_b: Vec<f32>,
    dw1_w: Vec<f32>, dw1_b: Vec<f32>,
    pw2_w: Vec<f32>, pw2_b: Vec<f32>,
    dw2_w: Vec<f32>, dw2_b: Vec<f32>,
    fc_w: Vec<f32>, fc_b: Vec<f32>,
}

impl MicroNet {
    fn from_bin(w_path: &str, lut_path: Option<&str>) -> Self {
        let bytes = fs::read(w_path).expect("read micro weights");
        assert_eq!(&bytes[0..4], b"MQV3");
        let mut r = Reader::new(&bytes[4..]);
        let (_ver, kid, nc, _tot) = (r.u32(), r.u32(), r.u32() as usize, r.u32());
        let (lut, br_w, br_b) = if kid == 4 {
            (Some(load_lut_i8(lut_path.expect("lut required for micro_quanv"))), None, None)
        } else {
            assert_eq!(kid, 5);
            let (bw0, bb0) = (r.vec_f32(16), r.vec_f32(4));
            let (bw1, bb1) = (r.vec_f32(16), r.vec_f32(4));
            let (bw2, bb2) = (r.vec_f32(16), r.vec_f32(4));
            let (bw3, bb3) = (r.vec_f32(36), r.vec_f32(4));
            (None, Some([bw0, bw1, bw2, bw3]), Some([bb0, bb1, bb2, bb3]))
        };
        Self {
            nc,
            weight_bytes: bytes.len() - 20,
            lut, br_w, br_b,
            stem0_s: r.vec_f32(16), stem0_sh: r.vec_f32(16),
            pw1_w: r.vec_f32(24 * 16), pw1_b: r.vec_f32(24),
            dw1_w: r.vec_f32(24 * 9), dw1_b: r.vec_f32(24),
            pw2_w: r.vec_f32(32 * 24), pw2_b: r.vec_f32(32),
            dw2_w: r.vec_f32(32 * 9), dw2_b: r.vec_f32(32),
            fc_w: r.vec_f32(nc * 32), fc_b: r.vec_f32(nc),
        }
    }
}

struct Workspace {
    x_norm: Vec<f32>,
    x_01: Vec<f32>,
    bins1: [usize; IMG_PIXELS],
    bins2: [usize; 4 * MS_HW],
    q1: Vec<f32>,
    f_c: Vec<f32>,
    b16_a: Vec<f32>,
    b16_b: Vec<f32>,
    b32_a: Vec<f32>,
    b32_b: Vec<f32>,
    b48: Vec<f32>,
    h64_a: Vec<f32>,
    h64_b: Vec<f32>,
    h64_c: Vec<f32>,
    p7_64: Vec<f32>,
    p7_64b: Vec<f32>,
    p7_96: Vec<f32>,
    p7_96b: Vec<f32>,
    p7_96s: Vec<f32>,
    p3_96: Vec<f32>,
    s2_map: Vec<f32>,
    // v2 head buffers
    h48_a: Vec<f32>,
    h48_b: Vec<f32>,
    h48_c: Vec<f32>,
    p7_48: Vec<f32>,
    p7_64v2: Vec<f32>,
    p7_64v2_b: Vec<f32>,
    p7_64v2_s: Vec<f32>,
    p3_64: Vec<f32>,
    // micro buffers
    m24_a: Vec<f32>,
    m24_b: Vec<f32>,
    m24_p7: Vec<f32>,
    m32_a: Vec<f32>,
    m32_b: Vec<f32>,
    gap: Vec<f32>,
}

impl Workspace {
    fn new() -> Self {
        Self {
            x_norm: vec![0.0; IMG_PIXELS],
            x_01: vec![0.0; IMG_PIXELS],
            bins1: [0; IMG_PIXELS],
            bins2: [0; 4 * MS_HW],
            q1: vec![0.0; 16 * MS_HW],
            f_c: vec![0.0; 16 * MS_HW],
            b16_a: vec![0.0; 16 * MS_HW],
            b16_b: vec![0.0; 16 * MS_HW],
            b32_a: vec![0.0; 32 * MS_HW],
            b32_b: vec![0.0; 32 * MS_HW],
            b48: vec![0.0; 48 * MS_HW],
            h64_a: vec![0.0; 64 * MS_HW],
            h64_b: vec![0.0; 64 * MS_HW],
            h64_c: vec![0.0; 64 * MS_HW],
            p7_64: vec![0.0; 64 * 49],
            p7_64b: vec![0.0; 64 * 49],
            p7_96: vec![0.0; 96 * 49],
            p7_96b: vec![0.0; 96 * 49],
            p7_96s: vec![0.0; 96 * 49],
            p3_96: vec![0.0; 96 * 9],
            s2_map: vec![0.0; 4 * MS_HW],
            h48_a: vec![0.0; 48 * MS_HW],
            h48_b: vec![0.0; 48 * MS_HW],
            h48_c: vec![0.0; 48 * MS_HW],
            p7_48: vec![0.0; 48 * 49],
            p7_64v2: vec![0.0; 64 * 49],
            p7_64v2_b: vec![0.0; 64 * 49],
            p7_64v2_s: vec![0.0; 64 * 49],
            p3_64: vec![0.0; 64 * 9],
            m24_a: vec![0.0; 24 * MS_HW],
            m24_b: vec![0.0; 24 * MS_HW],
            m24_p7: vec![0.0; 24 * 49],
            m32_a: vec![0.0; 32 * 49],
            m32_b: vec![0.0; 32 * 49],
            gap: vec![0.0; 96],
        }
    }

    #[inline]
    fn prep_input(&mut self, img_u8: &[u8]) {
        const INV_255: f32 = 1.0 / 255.0;
        for i in 0..IMG_PIXELS {
            let x = ((img_u8[i] as f32) * INV_255 - 0.5) * 2.0;
            self.x_norm[i] = x;
            self.x_01[i] = (x * 0.5 + 0.5).clamp(0.0, 1.0);
        }
    }
}

fn forward_v3_common_tail(h: &CommonV3Head, nc: usize, ws: &mut Workspace) -> [f32; 8] {
    conv1x1(&ws.b48, &h.head_stem_w, &h.head_stem_b, &mut ws.h64_a, 48, 64, MS_HW, true);
    // block1: DSResBlock(64, 64)
    conv_dw3x3(&ws.h64_a, &h.b1_dw1_w, None, &mut ws.h64_b, 64, MS_H, MS_W, 1, 1, false);
    conv1x1(&ws.h64_b, &h.b1_pw1_w, &h.b1_pw1_b, &mut ws.h64_c, 64, 64, MS_HW, true);
    conv_dw3x3(&ws.h64_c, &h.b1_dw2_w, None, &mut ws.h64_b, 64, MS_H, MS_W, 1, 1, false);
    conv1x1(&ws.h64_b, &h.b1_pw2_w, &h.b1_pw2_b, &mut ws.h64_c, 64, 64, MS_HW, false);
    for i in 0..(64 * MS_HW) {
        let v = ws.h64_c[i] + ws.h64_a[i];
        ws.h64_a[i] = if v > 0.0 { v } else { 0.0 };
    }
    maxpool2x2(&ws.h64_a, &mut ws.p7_64, 64, MS_H, MS_W);
    // block2: DSResBlock(64, 96)
    conv_dw3x3(&ws.p7_64, &h.b2_dw1_w, None, &mut ws.p7_64b, 64, 7, 7, 1, 1, false);
    conv1x1(&ws.p7_64b, &h.b2_pw1_w, &h.b2_pw1_b, &mut ws.p7_96, 64, 96, 49, true);
    conv_dw3x3(&ws.p7_96, &h.b2_dw2_w, None, &mut ws.p7_96b, 96, 7, 7, 1, 1, false);
    conv1x1(&ws.p7_96b, &h.b2_pw2_w, &h.b2_pw2_b, &mut ws.p7_96, 96, 96, 49, false);
    conv1x1(&ws.p7_64, &h.b2_skip_w, &h.b2_skip_b, &mut ws.p7_96s, 64, 96, 49, false);
    for i in 0..(96 * 49) {
        let v = ws.p7_96[i] + ws.p7_96s[i];
        ws.p7_96[i] = if v > 0.0 { v } else { 0.0 };
    }
    maxpool2x2(&ws.p7_96, &mut ws.p3_96, 96, 7, 7);
    gap_and_fc(&ws.p3_96, &h.fc_w, &h.fc_b, 96, 9, nc, &mut ws.gap)
}

fn run_v3_mixer_and_gate(h: &CommonV3Head, with_cdrop: bool, ws: &mut Workspace) {
    // q_mixer on q1 -> f_q in b16_b
    bn_affine(&mut ws.q1, &h.q_in_s, &h.q_in_sh, MS_HW, false);
    conv_dw3x3(&ws.q1, &h.qm_dw1_w, None, &mut ws.b16_a, 16, MS_H, MS_W, 1, 1, false);
    conv1x1(&ws.b16_a, &h.qm_pw1_w, &h.qm_pw1_b, &mut ws.b32_a, 16, 32, MS_HW, false);
    for v in ws.b32_a.iter_mut() {
        *v = gelu_exact(*v);
    }
    conv_dw3x3(&ws.b32_a, &h.qm_dw2_w, None, &mut ws.b32_b, 32, MS_H, MS_W, 2, 2, false);
    conv1x1(&ws.b32_b, &h.qm_pw2_w, &h.qm_pw2_b, &mut ws.b16_b, 32, 16, MS_HW, false);

    // c_stem -> norm_c -> stabilize
    conv_c_stem(&ws.x_norm, &h.c_stem_w, &h.c_stem_b, &mut ws.f_c);
    let c_drop = stabilize_f_c_and_cdrop(&ws.x_01, &mut ws.f_c, with_cdrop);

    // gate input cat[f_c, f_q] in b32_a
    ws.b32_a[..16 * MS_HW].copy_from_slice(&ws.f_c);
    ws.b32_a[16 * MS_HW..].copy_from_slice(&ws.b16_b);
    bn_affine(&mut ws.b32_a, &h.gate_bn_s, &h.gate_bn_sh, MS_HW, false);
    conv_dw3x3(&ws.b32_a, &h.gate_dw_w, None, &mut ws.b32_b, 32, MS_H, MS_W, 1, 1, false);
    conv1x1(&ws.b32_b, &h.gate_pw_w, &h.gate_pw_b, &mut ws.b16_a, 32, 16, MS_HW, false);

    // Symmetric variance-preserving fusion -> b48[0..32*MS_HW], keeping w_q * f_q in b16_b
    const SQRT2: f32 = std::f32::consts::SQRT_2;
    const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
    for i in 0..(16 * MS_HW) {
        let alpha = sigmoid(ws.b16_a[i]);
        let theta = HALF_PI * alpha;
        let wc = SQRT2 * theta.cos() * c_drop;
        let wq = SQRT2 * theta.sin();
        let wq_fq = wq * ws.b16_b[i];
        ws.b48[i] = wc * ws.f_c[i];
        ws.b48[16 * MS_HW + i] = wq_fq;
        ws.b16_b[i] = wq_fq;
    }
}

fn infer_quanv_v3(net: &QuanvV3Net, img_u8: &[u8], ws: &mut Workspace) -> [f32; 8] {
    ws.prep_input(img_u8);
    qlut_stage1_extract(&ws.x_01, &net.lut, &mut ws.q1, &mut ws.bins1);
    run_v3_mixer_and_gate(&net.h, true, ws);
    // Stage-2 Q-LUT on w_q * f_q (ws.b16_b)
    conv1x1(&ws.b16_b, &net.h.s2_w0, &net.h.s2_b0, &mut ws.s2_map, 16, 4, MS_HW, false);
    {
        let q2_slice = &mut ws.b48[32 * MS_HW..];
        qlut_stage2_extract(&ws.s2_map, &net.lut, q2_slice, &mut ws.bins2);
        bn_affine(q2_slice, &net.h.s2_w1, &net.h.s2_b1, MS_HW, false);
    }
    forward_v3_common_tail(&net.h, net.nc, ws)
}

fn infer_ctrl_v3(net: &CtrlV3LightNet, img_u8: &[u8], ws: &mut Workspace) -> [f32; 8] {
    ws.prep_input(img_u8);
    conv_classical_branches(&ws.x_norm, &net.br_w, &net.br_b, &mut ws.q1);
    run_v3_mixer_and_gate(&net.h, false, ws);
    // Classical Stage-2: s2_proj.0+1 + ReLU -> grouped Conv3x3 + BN
    conv1x1(&ws.b16_b, &net.h.s2_w0, &net.h.s2_b0, &mut ws.s2_map, 16, 4, MS_HW, true);
    conv_grouped3x3(
        &ws.s2_map,
        &net.h.s2_w1,
        &net.h.s2_b1,
        &mut ws.b48[32 * MS_HW..],
        4,
        16,
        MS_H,
        MS_W,
    );
    forward_v3_common_tail(&net.h, net.nc, ws)
}

fn infer_ctrl_v2(net: &CtrlV2FullNet, img_u8: &[u8], ws: &mut Workspace) -> [f32; 8] {
    ws.prep_input(img_u8);
    conv_c_stem(&ws.x_norm, &net.c_stem_w, &net.c_stem_b, &mut ws.f_c);
    let _ = stabilize_f_c_and_cdrop(&ws.x_01, &mut ws.f_c, false);

    conv_classical_branches(&ws.x_norm, &net.br_w, &net.br_b, &mut ws.q1);
    bn_affine(&mut ws.q1, &net.mx_bn0_s, &net.mx_bn0_sh, MS_HW, false);
    conv3x3(&ws.q1, &net.mx_c1_w, &net.mx_c1_b, &mut ws.b32_a, 16, 32, MS_H, MS_W, 1, 1, Act::Gelu);
    conv3x3(&ws.b32_a, &net.mx_c2_w, &net.mx_c2_b, &mut ws.b32_b, 32, 32, MS_H, MS_W, 2, 2, Act::Gelu);
    conv1x1(&ws.b32_b, &net.mx_c3_w, &net.mx_c3_b, &mut ws.b16_b, 32, 16, MS_HW, false);

    ws.b32_a[..16 * MS_HW].copy_from_slice(&ws.f_c);
    ws.b32_a[16 * MS_HW..].copy_from_slice(&ws.b16_b);
    bn_affine(&mut ws.b32_a, &net.gate_bn_s, &net.gate_bn_sh, MS_HW, false);
    conv3x3(&ws.b32_a, &net.gate_c1_w, &net.gate_c1_b, &mut ws.b32_b, 32, 32, MS_H, MS_W, 1, 1, Act::Gelu);
    conv1x1(&ws.b32_b, &net.gate_c2_w, &net.gate_c2_b, &mut ws.b16_a, 32, 16, MS_HW, false);

    // f_fused in b32_a = cat[f_c, sigmoid(alpha) * f_ms]
    ws.b32_a[..16 * MS_HW].copy_from_slice(&ws.f_c);
    for i in 0..(16 * MS_HW) {
        ws.b32_a[16 * MS_HW + i] = sigmoid(ws.b16_a[i]) * ws.b16_b[i];
    }

    // WideResidualHead
    bn_affine_into(&ws.b32_a, &mut ws.b32_b, &net.head_pre_s, &net.head_pre_sh, MS_HW, true);
    conv3x3(&ws.b32_b, &net.head_stem_w, &net.head_stem_b, &mut ws.h48_a, 32, 48, MS_H, MS_W, 1, 1, Act::Relu);
    // block1
    conv3x3(&ws.h48_a, &net.b1_c1_w, &net.b1_c1_b, &mut ws.h48_b, 48, 48, MS_H, MS_W, 1, 1, Act::Relu);
    conv3x3(&ws.h48_b, &net.b1_c2_w, &net.b1_c2_b, &mut ws.h48_c, 48, 48, MS_H, MS_W, 1, 1, Act::None);
    for i in 0..(48 * MS_HW) {
        let v = ws.h48_c[i] + ws.h48_a[i];
        ws.h48_a[i] = if v > 0.0 { v } else { 0.0 };
    }
    maxpool2x2(&ws.h48_a, &mut ws.p7_48, 48, MS_H, MS_W);
    // block2
    conv3x3(&ws.p7_48, &net.b2_c1_w, &net.b2_c1_b, &mut ws.p7_64v2, 48, 64, 7, 7, 1, 1, Act::Relu);
    conv3x3(&ws.p7_64v2, &net.b2_c2_w, &net.b2_c2_b, &mut ws.p7_64v2_b, 64, 64, 7, 7, 1, 1, Act::None);
    conv1x1(&ws.p7_48, &net.b2_skip_w, &net.b2_skip_b, &mut ws.p7_64v2_s, 48, 64, 49, false);
    for i in 0..(64 * 49) {
        let v = ws.p7_64v2_b[i] + ws.p7_64v2_s[i];
        ws.p7_64v2[i] = if v > 0.0 { v } else { 0.0 };
    }
    maxpool2x2(&ws.p7_64v2, &mut ws.p3_64, 64, 7, 7);
    gap_and_fc(&ws.p3_64, &net.fc_w, &net.fc_b, 64, 9, net.nc, &mut ws.gap)
}

fn infer_micro(net: &MicroNet, img_u8: &[u8], ws: &mut Workspace) -> [f32; 8] {
    ws.prep_input(img_u8);
    if let Some(ref lut) = net.lut {
        qlut_stage1_extract(&ws.x_01, lut, &mut ws.q1, &mut ws.bins1);
    } else if let (Some(ref bw), Some(ref bb)) = (&net.br_w, &net.br_b) {
        conv_classical_branches(&ws.x_norm, bw, bb, &mut ws.q1);
    }
    bn_affine(&mut ws.q1, &net.stem0_s, &net.stem0_sh, MS_HW, false);
    conv1x1(&ws.q1, &net.pw1_w, &net.pw1_b, &mut ws.m24_a, 16, 24, MS_HW, true);
    conv_dw3x3(&ws.m24_a, &net.dw1_w, Some(&net.dw1_b), &mut ws.m24_b, 24, MS_H, MS_W, 1, 1, true);
    avgpool2x2(&ws.m24_b, &mut ws.m24_p7, 24, MS_H, MS_W);
    conv1x1(&ws.m24_p7, &net.pw2_w, &net.pw2_b, &mut ws.m32_a, 24, 32, 49, true);
    conv_dw3x3(&ws.m32_a, &net.dw2_w, Some(&net.dw2_b), &mut ws.m32_b, 32, 7, 7, 1, 1, true);
    gap_and_fc(&ws.m32_b, &net.fc_w, &net.fc_b, 32, 49, net.nc, &mut ws.gap)
}

struct V3VerifyBundle {
    n: usize,
    imgs_clean: Vec<u8>,
    imgs_s20: Vec<u8>,
    labels: Vec<u8>,
    ref_clean: Vec<Vec<f32>>,
    ref_s20: Vec<Vec<f32>>,
}

impl V3VerifyBundle {
    fn from_bin(path: &str) -> Self {
        let bytes = fs::read(path).expect("read verify bin");
        assert_eq!(&bytes[0..4], b"MQR3");
        let mut r = Reader::new(&bytes[4..]);
        let (_ver, n, h, w, n_models) = (r.u32(), r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        assert_eq!(h * w, IMG_PIXELS);
        let p = &r.buf[r.pos..];
        let mut off = 0usize;
        let imgs_clean = p[off..off + n * IMG_PIXELS].to_vec(); off += n * IMG_PIXELS;
        let imgs_s20 = p[off..off + n * IMG_PIXELS].to_vec(); off += n * IMG_PIXELS;
        let labels = p[off..off + n].to_vec(); off += n;

        let mut read_logits = || -> Vec<Vec<f32>> {
            let mut out = Vec::with_capacity(n_models);
            for _ in 0..n_models {
                let mut v = Vec::with_capacity(n * 2);
                for _ in 0..(n * 2) {
                    let b = &p[off..off + 4];
                    off += 4;
                    v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
                out.push(v);
            }
            out
        };
        let ref_clean = read_logits();
        let ref_s20 = read_logits();
        Self { n, imgs_clean, imgs_s20, labels, ref_clean, ref_s20 }
    }
}

struct BenchStats {
    name: &'static str,
    acc_clean: f64,
    py_acc_clean: f64,
    diff_clean: f32,
    acc_s20: f64,
    py_acc_s20: f64,
    diff_s20: f32,
    agree_pct: f64,
    e2e_us: f64,
    e2e_fps: f64,
    lut_us: f64,
    weight_kib: f64,
    lut_kib: f64,
    total_kib: f64,
    l2_pct: f64,
}

fn eval_and_bench<F>(
    name: &'static str,
    bundle: &V3VerifyBundle,
    m_idx: usize,
    weight_bytes: usize,
    lut_bytes: usize,
    ws: &mut Workspace,
    mut run_fn: F,
    qlut_only_us: f64,
) -> BenchStats
where
    F: FnMut(&[u8], &mut Workspace) -> [f32; 8],
{
    let n = bundle.n;
    let eval_split = |imgs: &[u8], refs: &[f32], f: &mut F, w: &mut Workspace| -> (f64, f64, f32, usize) {
        let mut r_ok = 0usize;
        let mut p_ok = 0usize;
        let mut agree = 0usize;
        let mut max_d = 0.0f32;
        for s in 0..n {
            let lg = f(&imgs[s * IMG_PIXELS..(s + 1) * IMG_PIXELS], w);
            let r_pred = if lg[1] > lg[0] { 1usize } else { 0usize };
            let r0 = refs[s * 2];
            let r1 = refs[s * 2 + 1];
            let p_pred = if r1 > r0 { 1usize } else { 0usize };
            let y = bundle.labels[s] as usize;
            if r_pred == y { r_ok += 1; }
            if p_pred == y { p_ok += 1; }
            if r_pred == p_pred { agree += 1; }
            max_d = max_d.max((lg[0] - r0).abs()).max((lg[1] - r1).abs());
        }
        (100.0 * r_ok as f64 / n as f64, 100.0 * p_ok as f64 / n as f64, max_d, agree)
    };

    let (acc_clean, py_acc_clean, diff_clean, agree_c) =
        eval_split(&bundle.imgs_clean, &bundle.ref_clean[m_idx], &mut run_fn, ws);
    let (acc_s20, py_acc_s20, diff_s20, agree_s) =
        eval_split(&bundle.imgs_s20, &bundle.ref_s20[m_idx], &mut run_fn, ws);
    let agree_pct = 100.0 * (agree_c + agree_s) as f64 / (2 * n) as f64;

    // Warmup 3 full dataset passes, then measure 8 full passes (8 * 624 = 4,992 E2E inferences)
    for _ in 0..3 {
        for s in 0..n {
            let lg = run_fn(black_box(&bundle.imgs_clean[s * IMG_PIXELS..(s + 1) * IMG_PIXELS]), ws);
            black_box(lg);
        }
    }
    let passes = 8usize;
    let t0 = Instant::now();
    for _ in 0..passes {
        for s in 0..n {
            let lg = run_fn(black_box(&bundle.imgs_clean[s * IMG_PIXELS..(s + 1) * IMG_PIXELS]), ws);
            black_box(lg);
        }
    }
    let e2e_us = t0.elapsed().as_secs_f64() * 1e6 / ((passes * n) as f64);
    let e2e_fps = 1e6 / e2e_us;
    let weight_kib = weight_bytes as f64 / 1024.0;
    let lut_kib = lut_bytes as f64 / 1024.0;
    let total_kib = weight_kib + lut_kib;
    let l2_pct = 100.0 * total_kib / 2048.0;

    println!(
        "  [{:<14}] Clean={:5.2}% (diff={:.2e}) | s0.20={:5.2}% (diff={:.2e}) | Agree={:5.1}% | E2E={:7.2} us ({:7.1} FPS) | L2={:6.1} KiB ({:4.1}%)",
        name, acc_clean, diff_clean, acc_s20, diff_s20, agree_pct, e2e_us, e2e_fps, total_kib, l2_pct
    );

    BenchStats {
        name, acc_clean, py_acc_clean, diff_clean, acc_s20, py_acc_s20, diff_s20,
        agree_pct, e2e_us, e2e_fps, lut_us: qlut_only_us,
        weight_kib, lut_kib, total_kib, l2_pct,
    }
}

pub fn run_v3_suite(args: &[String]) {
    let mut verify_path = "data/processed/pneumoniamnist_v3_rust_verify.bin".to_string();
    let mut json_out = "reports/medquanv_cli_v3_verification.json".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--verify" && i + 1 < args.len() {
            verify_path = args[i + 1].clone();
            i += 1;
        } else if args[i] == "--json-out" && i + 1 < args.len() {
            json_out = args[i + 1].clone();
            i += 1;
        }
        i += 1;
    }

    let bundle = V3VerifyBundle::from_bin(&verify_path);
    let quanv_v3 = QuanvV3Net::from_bin(
        "weights/v3_quanv_pneumoniamnist_s42.bin",
        "weights/v3_pneumoniamnist_s42_int8_shift.bin",
    );
    let ctrl_v3 = CtrlV3LightNet::from_bin("weights/v3_ctrl_light_pneumoniamnist_s42.bin");
    let ctrl_v2 = CtrlV2FullNet::from_bin("weights/v2_ctrl_full_pneumoniamnist_s42.bin");
    let micro_q = MicroNet::from_bin(
        "weights/v3_micro_quanv_pneumoniamnist_s42.bin",
        Some("weights/v3_pneumoniamnist_s42_micro_int8_shift.bin"),
    );
    let micro_c = MicroNet::from_bin("weights/v3_micro_ctrl_pneumoniamnist_s42.bin", None);

    let mut ws = Workspace::new();

    // Measure pure Q-LUT Stage-1 time per image (microseconds)
    ws.prep_input(&bundle.imgs_clean[0..IMG_PIXELS]);
    let t_lut0 = Instant::now();
    for _ in 0..20000 {
        qlut_stage1_extract(black_box(&ws.x_01), &quanv_v3.lut, black_box(&mut ws.q1), &mut ws.bins1);
    }
    let stage1_lut_us = t_lut0.elapsed().as_secs_f64() * 1e6 / 20000.0;

    let arch_name = if cfg!(target_arch = "aarch64") {
        "ARM64 (aarch64 NEON - Cortex-A57 Target)"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64 (AVX2/FMA/SSSE3 Native)"
    } else {
        "Generic Scalar"
    };
    println!("=================================================================================");
    println!("  medquanv-cli v3 Suite | Pure-Rust End-to-End Parity & Latency (N={} x 2 regimes)", bundle.n);
    println!("  Target ISA: {} | Stage-1 4-scale INT8 Q-LUT: {:.3} us/img", arch_name, stage1_lut_us);
    println!("=================================================================================");

    let results = vec![
        eval_and_bench(
            "quanv_v3_int8", &bundle, 0, quanv_v3.weight_bytes, LUT_BYTES, &mut ws,
            |img, w| infer_quanv_v3(&quanv_v3, img, w), stage1_lut_us,
        ),
        eval_and_bench(
            "ctrl_v3_light", &bundle, 1, ctrl_v3.weight_bytes, 0, &mut ws,
            |img, w| infer_ctrl_v3(&ctrl_v3, img, w), 0.0,
        ),
        eval_and_bench(
            "ctrl_v2_full", &bundle, 2, ctrl_v2.weight_bytes, 0, &mut ws,
            |img, w| infer_ctrl_v2(&ctrl_v2, img, w), 0.0,
        ),
        eval_and_bench(
            "micro_quanv", &bundle, 3, micro_q.weight_bytes, LUT_BYTES, &mut ws,
            |img, w| infer_micro(&micro_q, img, w), stage1_lut_us,
        ),
        eval_and_bench(
            "micro_ctrl", &bundle, 4, micro_c.weight_bytes, 0, &mut ws,
            |img, w| infer_micro(&micro_c, img, w), 0.0,
        ),
    ];

    let mut json = String::from("{\n  \"arch\": \"");
    json.push_str(arch_name);
    json.push_str(&format!("\",\n  \"n_samples\": {},\n  \"stage1_qlut_int8_us\": {:.4},\n  \"models\": {{\n", bundle.n, stage1_lut_us));
    for (idx, r) in results.iter().enumerate() {
        json.push_str(&format!(
            "    \"{}\": {{\n      \"rust_acc_clean\": {:.4}, \"pytorch_acc_clean\": {:.4}, \"max_logit_diff_clean\": {:.6e},\n      \"rust_acc_s20\": {:.4}, \"pytorch_acc_s20\": {:.4}, \"max_logit_diff_s20\": {:.6e},\n      \"prediction_agreement_pct\": {:.2}, \"e2e_us\": {:.3}, \"e2e_fps\": {:.2},\n      \"qlut_stage1_us\": {:.4}, \"weight_kib\": {:.2}, \"lut_kib\": {:.2}, \"total_kib\": {:.2}, \"l2_cache_pct\": {:.2}\n    }}{}\n",
            r.name, r.acc_clean, r.py_acc_clean, r.diff_clean, r.acc_s20, r.py_acc_s20, r.diff_s20,
            r.agree_pct, r.e2e_us, r.e2e_fps, r.lut_us, r.weight_kib, r.lut_kib, r.total_kib, r.l2_pct,
            if idx + 1 < results.len() { "," } else { "" }
        ));
    }
    json.push_str("  }\n}\n");
    fs::write(&json_out, json).expect("write v3 verification json");
    println!("[DONE] Saved v3 Rust suite telemetry -> {}", json_out);
}
