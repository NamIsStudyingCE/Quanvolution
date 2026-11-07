use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::arch::x86_64::*;

const NUM_STATES: usize = 65536;
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_PIXELS: usize = IMG_H * IMG_W; // 784
const OUT_H: usize = 14;
const OUT_W: usize = 14;
const NUM_PATCHES: usize = OUT_H * OUT_W; // 196 (divisible by 4: 49 groups of 4 patches)
const OUT_ELEMS: usize = NUM_PATCHES * 4; // 784

#[repr(C, align(64))]
pub struct AlignedBytes<const N: usize>(pub [u8; N]);

// Bảng INT8 (256 KiB = 262,144 bytes) căn chỉnh 64-byte Cache Line trong .rdata
static LUT_INT8_GRID: AlignedBytes<{ NUM_STATES * 4 }> = AlignedBytes(*include_bytes!(
    "../../weights/qlut_strongly_L2_4bit_int8_grid.bin"
));
static LUT_INT8_SHIFT: AlignedBytes<{ NUM_STATES * 4 }> = AlignedBytes(*include_bytes!(
    "../../weights/qlut_strongly_L2_4bit_int8_shift.bin"
));

// Bảng FP32 (1,024 KiB = 1,048,576 bytes) căn chỉnh 64-byte Cache Line trong .rdata
static LUT_FP32_GRID: AlignedBytes<{ NUM_STATES * 16 }> = AlignedBytes(*include_bytes!(
    "../../weights/qlut_strongly_L2_4bit_fp32_grid.bin"
));
static LUT_FP32_SHIFT: AlignedBytes<{ NUM_STATES * 16 }> = AlignedBytes(*include_bytes!(
    "../../weights/qlut_strongly_L2_4bit_fp32_shift.bin"
));

// Bảng tiền tính cho Nội suy 4D trên u8 (256 mức): (lo_idx, frac)
#[repr(C, align(64))]
struct InterpPrecomp {
    lo0: [usize; 256],
    lo1: [usize; 256],
    lo2: [usize; 256],
    lo3: [usize; 256],
    frac: [f32; 256],
}

static INTERP_TABLE: InterpPrecomp = {
    let mut t = InterpPrecomp {
        lo0: [0; 256],
        lo1: [0; 256],
        lo2: [0; 256],
        lo3: [0; 256],
        frac: [0.0; 256],
    };
    let mut p = 0usize;
    while p < 256 {
        let s = (p as f32 / 255.0) * 15.0;
        let mut lo = s as usize;
        if lo > 14 {
            lo = 14;
        }
        let f = s - (lo as f32);
        t.lo0[p] = lo << 12;
        t.lo1[p] = lo << 8;
        t.lo2[p] = lo << 4;
        t.lo3[p] = lo;
        t.frac[p] = f;
        p += 1;
    }
    t
};

// Bảng ánh xạ u8 -> 4-bit làm tròn chuẩn round(p * 15 / 255) đã dịch sẵn bit (tổng 2 KiB trong L1 Cache)
#[repr(C, align(64))]
struct GridIndexMap {
    s12: [usize; 256],
    s8: [usize; 256],
    s4: [usize; 256],
    s0: [usize; 256],
}

static GRID_MAP: GridIndexMap = {
    let mut m = GridIndexMap {
        s12: [0; 256],
        s8: [0; 256],
        s4: [0; 256],
        s0: [0; 256],
    };
    let mut p = 0usize;
    while p < 256 {
        // 255 = 15 * 17 -> round(p * 15 / 255) = round(p / 17) = (p + 8) / 17
        let q = (p + 8) / 17;
        m.s12[p] = q << 12;
        m.s8[p] = q << 8;
        m.s4[p] = q << 4;
        m.s0[p] = q;
        p += 1;
    }
    m
};

#[inline(always)]
unsafe fn patch_idx_shift(img: *const u8, r2: usize, c2: usize) -> usize {
    let row0 = img.add(r2 * IMG_W + c2);
    let row1 = row0.add(IMG_W);
    let p0 = *row0 as usize;
    let p1 = *row0.add(1) as usize;
    let p2 = *row1 as usize;
    let p3 = *row1.add(1) as usize;
    ((p0 & 0xF0) << 8) | ((p1 & 0xF0) << 4) | (p2 & 0xF0) | (p3 >> 4)
}

#[inline(always)]
unsafe fn patch_idx_grid(img: *const u8, r2: usize, c2: usize) -> usize {
    let row0 = img.add(r2 * IMG_W + c2);
    let row1 = row0.add(IMG_W);
    let p0 = *row0 as usize;
    let p1 = *row0.add(1) as usize;
    let p2 = *row1 as usize;
    let p3 = *row1.add(1) as usize;
    (*GRID_MAP.s12.get_unchecked(p0))
        | (*GRID_MAP.s8.get_unchecked(p1))
        | (*GRID_MAP.s4.get_unchecked(p2))
        | (*GRID_MAP.s0.get_unchecked(p3))
}

// ============================================================================
// 1. KERNEL INT8 NHWC (Channels-Last: 14 x 14 x 4) - 1 lệnh load/store u32 mỗi patch
// ============================================================================
#[inline(always)]
pub unsafe fn kernel_int8_nhwc_single(img: *const u8, out_u32: *mut u32, lut_u32: *const u32, use_shift: bool) {
    let mut patch_i = 0usize;
    if use_shift {
        for r in 0..OUT_H {
            let r2 = r * 2;
            // Unroll 14 patches along the row
            for c in 0..OUT_W {
                let idx = patch_idx_shift(img, r2, c * 2);
                *out_u32.add(patch_i) = *lut_u32.add(idx);
                patch_i += 1;
            }
        }
    } else {
        for r in 0..OUT_H {
            let r2 = r * 2;
            for c in 0..OUT_W {
                let idx = patch_idx_grid(img, r2, c * 2);
                *out_u32.add(patch_i) = *lut_u32.add(idx);
                patch_i += 1;
            }
        }
    }
}

// ============================================================================
// 2. KERNEL INT8 NCHW (Channels-First: 4 x 14 x 14) - SIMD SSSE3 pshufb Transpose 4x4
// ============================================================================
#[inline(always)]
pub unsafe fn kernel_int8_nchw_single(img: *const u8, out_i8: *mut u8, lut_u32: *const u32, use_shift: bool) {
    // 196 patches = 49 groups of 4 patches
    // Mask pshufb chuyển vị ma trận 4 patches x 4 channels (16 bytes) trong 1 chu kỳ:
    let shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let out_c0 = out_i8 as *mut u32;
    let out_c1 = out_i8.add(NUM_PATCHES) as *mut u32;
    let out_c2 = out_i8.add(NUM_PATCHES * 2) as *mut u32;
    let out_c3 = out_i8.add(NUM_PATCHES * 3) as *mut u32;

    // Quét 196 patches theo nhóm 4 (49 nhóm)
    let mut indices = [0usize; NUM_PATCHES];
    let mut p = 0usize;
    if use_shift {
        for r in 0..OUT_H {
            let r2 = r * 2;
            for c in 0..OUT_W {
                *indices.get_unchecked_mut(p) = patch_idx_shift(img, r2, c * 2);
                p += 1;
            }
        }
    } else {
        for r in 0..OUT_H {
            let r2 = r * 2;
            for c in 0..OUT_W {
                *indices.get_unchecked_mut(p) = patch_idx_grid(img, r2, c * 2);
                p += 1;
            }
        }
    }

    for g in 0..49 {
        let base = g * 4;
        let v0 = *lut_u32.add(*indices.get_unchecked(base)) as i32;
        let v1 = *lut_u32.add(*indices.get_unchecked(base + 1)) as i32;
        let v2 = *lut_u32.add(*indices.get_unchecked(base + 2)) as i32;
        let v3 = *lut_u32.add(*indices.get_unchecked(base + 3)) as i32;

        let vec = _mm_setr_epi32(v0, v1, v2, v3);
        let tr = _mm_shuffle_epi8(vec, shuf_mask);

        *out_c0.add(g) = _mm_cvtsi128_si32(tr) as u32;
        *out_c1.add(g) = _mm_extract_epi32(tr, 1) as u32;
        *out_c2.add(g) = _mm_extract_epi32(tr, 2) as u32;
        *out_c3.add(g) = _mm_extract_epi32(tr, 3) as u32;
    }
}

// ============================================================================
// 3. KERNEL FP32 NCHW (4 x 14 x 14 float32) - SIMD 128-bit Load + 4x4 Transpose
// ============================================================================
#[inline(always)]
pub unsafe fn kernel_fp32_nchw_single(img: *const u8, out_f32: *mut f32, lut_f32: *const f32, use_shift: bool) {
    let out_c0 = out_f32;
    let out_c1 = out_f32.add(NUM_PATCHES);
    let out_c2 = out_f32.add(NUM_PATCHES * 2);
    let out_c3 = out_f32.add(NUM_PATCHES * 3);

    let mut indices = [0usize; NUM_PATCHES];
    let mut p = 0usize;
    if use_shift {
        for r in 0..OUT_H {
            let r2 = r * 2;
            for c in 0..OUT_W {
                *indices.get_unchecked_mut(p) = patch_idx_shift(img, r2, c * 2);
                p += 1;
            }
        }
    } else {
        for r in 0..OUT_H {
            let r2 = r * 2;
            for c in 0..OUT_W {
                *indices.get_unchecked_mut(p) = patch_idx_grid(img, r2, c * 2);
                p += 1;
            }
        }
    }

    for g in 0..49 {
        let base = g * 4;
        let mut r0 = _mm_load_ps(lut_f32.add(*indices.get_unchecked(base) * 4));
        let mut r1 = _mm_load_ps(lut_f32.add(*indices.get_unchecked(base + 1) * 4));
        let mut r2 = _mm_load_ps(lut_f32.add(*indices.get_unchecked(base + 2) * 4));
        let mut r3 = _mm_load_ps(lut_f32.add(*indices.get_unchecked(base + 3) * 4));

        _MM_TRANSPOSE4_PS(&mut r0, &mut r1, &mut r2, &mut r3);

        let offset = g * 4;
        _mm_storeu_ps(out_c0.add(offset), r0);
        _mm_storeu_ps(out_c1.add(offset), r1);
        _mm_storeu_ps(out_c2.add(offset), r2);
        _mm_storeu_ps(out_c3.add(offset), r3);
    }
}

// ============================================================================
// 4. KERNEL 4D MULTILINEAR INTERPOLATION (SIMD FMA Separable Reduction)
// ============================================================================
#[inline(always)]
unsafe fn lerp_m128(a: __m128, b: __m128, t: __m128) -> __m128 {
    // a + t * (b - a) bằng lệnh FMA đơn chu kỳ
    _mm_fmadd_ps(t, _mm_sub_ps(b, a), a)
}

#[inline(always)]
unsafe fn interp4d_patch_simd(lut_f32: *const f32, p0: usize, p1: usize, p2: usize, p3: usize) -> __m128 {
    let base = (*INTERP_TABLE.lo0.get_unchecked(p0))
        | (*INTERP_TABLE.lo1.get_unchecked(p1))
        | (*INTERP_TABLE.lo2.get_unchecked(p2))
        | (*INTERP_TABLE.lo3.get_unchecked(p3));

    let f0 = _mm_set1_ps(*INTERP_TABLE.frac.get_unchecked(p0));
    let f1 = _mm_set1_ps(*INTERP_TABLE.frac.get_unchecked(p1));
    let f2 = _mm_set1_ps(*INTERP_TABLE.frac.get_unchecked(p2));
    let f3 = _mm_set1_ps(*INTERP_TABLE.frac.get_unchecked(p3));

    // Khử chiều thứ 3 (stride = 1 trạng thái = 4 floats)
    macro_rules! load_lerp3 {
        ($off:expr) => {{
            let ptr = lut_f32.add((base + $off) * 4);
            let v0 = _mm_load_ps(ptr);
            let v1 = _mm_load_ps(ptr.add(4));
            lerp_m128(v0, v1, f3)
        }};
    }

    // 8 cặp dọc theo trục 3
    let c000 = load_lerp3!(0);
    let c001 = load_lerp3!(16);
    let c010 = load_lerp3!(256);
    let c011 = load_lerp3!(272);
    let c100 = load_lerp3!(4096);
    let c101 = load_lerp3!(4112);
    let c110 = load_lerp3!(4352);
    let c111 = load_lerp3!(4368);

    // Khử chiều thứ 2
    let c00 = lerp_m128(c000, c001, f2);
    let c01 = lerp_m128(c010, c011, f2);
    let c10 = lerp_m128(c100, c101, f2);
    let c11 = lerp_m128(c110, c111, f2);

    // Khử chiều thứ 1
    let c0 = lerp_m128(c00, c01, f1);
    let c1 = lerp_m128(c10, c11, f1);

    // Khử chiều thứ 0
    lerp_m128(c0, c1, f0)
}

#[inline(always)]
pub unsafe fn kernel_interp4d_nchw_single(img: *const u8, out_f32: *mut f32, lut_f32: *const f32) {
    let out_c0 = out_f32;
    let out_c1 = out_f32.add(NUM_PATCHES);
    let out_c2 = out_f32.add(NUM_PATCHES * 2);
    let out_c3 = out_f32.add(NUM_PATCHES * 3);

    let mut coords = [(0usize, 0usize); NUM_PATCHES];
    let mut p = 0usize;
    for r in 0..OUT_H {
        for c in 0..OUT_W {
            *coords.get_unchecked_mut(p) = (r * 2, c * 2);
            p += 1;
        }
    }

    for g in 0..49 {
        let base = g * 4;
        macro_rules! eval_patch {
            ($k:expr) => {{
                let (r2, c2) = *coords.get_unchecked(base + $k);
                let row0 = img.add(r2 * IMG_W + c2);
                let row1 = row0.add(IMG_W);
                interp4d_patch_simd(
                    lut_f32,
                    *row0 as usize,
                    *row0.add(1) as usize,
                    *row1 as usize,
                    *row1.add(1) as usize,
                )
            }};
        }
        let mut r0 = eval_patch!(0);
        let mut r1 = eval_patch!(1);
        let mut r2 = eval_patch!(2);
        let mut r3 = eval_patch!(3);

        _MM_TRANSPOSE4_PS(&mut r0, &mut r1, &mut r2, &mut r3);

        let offset = g * 4;
        _mm_storeu_ps(out_c0.add(offset), r0);
        _mm_storeu_ps(out_c1.add(offset), r1);
        _mm_storeu_ps(out_c2.add(offset), r2);
        _mm_storeu_ps(out_c3.add(offset), r3);
    }
}

// ============================================================================
// PYTHON BINDINGS (Zero-Copy NumPy & Direct Tensor Pointer API)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (imgs, out, use_shift=false, parallel=false))]
fn forward_int8_nhwc(
    py: Python<'_>,
    imgs: PyReadonlyArrayDyn<'_, u8>,
    mut out: PyReadwriteArrayDyn<'_, i8>,
    use_shift: bool,
    parallel: bool,
) -> PyResult<()> {
    let in_slice = imgs.as_slice()?;
    let out_slice = out.as_slice_mut()?;
    let b = in_slice.len() / IMG_PIXELS;
    assert_eq!(in_slice.len(), b * IMG_PIXELS);
    assert_eq!(out_slice.len(), b * OUT_ELEMS);

    let in_addr = in_slice.as_ptr() as usize;
    let out_addr = out_slice.as_mut_ptr() as usize;
    let lut_addr = if use_shift {
        LUT_INT8_SHIFT.0.as_ptr() as usize
    } else {
        LUT_INT8_GRID.0.as_ptr() as usize
    };

    py.allow_threads(move || unsafe {
        if parallel && b > 1 {
            (0..b).into_par_iter().for_each(|i| {
                let lut_u32 = lut_addr as *const u32;
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut u32).add(i * NUM_PATCHES);
                kernel_int8_nhwc_single(img_p, out_p, lut_u32, use_shift);
            });
        } else {
            let lut_u32 = lut_addr as *const u32;
            for i in 0..b {
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut u32).add(i * NUM_PATCHES);
                kernel_int8_nhwc_single(img_p, out_p, lut_u32, use_shift);
            }
        }
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (imgs, out, use_shift=false, parallel=false))]
fn forward_int8_nchw(
    py: Python<'_>,
    imgs: PyReadonlyArrayDyn<'_, u8>,
    mut out: PyReadwriteArrayDyn<'_, i8>,
    use_shift: bool,
    parallel: bool,
) -> PyResult<()> {
    let in_slice = imgs.as_slice()?;
    let out_slice = out.as_slice_mut()?;
    let b = in_slice.len() / IMG_PIXELS;
    assert_eq!(in_slice.len(), b * IMG_PIXELS);
    assert_eq!(out_slice.len(), b * OUT_ELEMS);

    let in_addr = in_slice.as_ptr() as usize;
    let out_addr = out_slice.as_mut_ptr() as usize;
    let lut_addr = if use_shift {
        LUT_INT8_SHIFT.0.as_ptr() as usize
    } else {
        LUT_INT8_GRID.0.as_ptr() as usize
    };

    py.allow_threads(move || unsafe {
        if parallel && b > 1 {
            (0..b).into_par_iter().for_each(|i| {
                let lut_u32 = lut_addr as *const u32;
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut u8).add(i * OUT_ELEMS);
                kernel_int8_nchw_single(img_p, out_p, lut_u32, use_shift);
            });
        } else {
            let lut_u32 = lut_addr as *const u32;
            for i in 0..b {
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut u8).add(i * OUT_ELEMS);
                kernel_int8_nchw_single(img_p, out_p, lut_u32, use_shift);
            }
        }
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (imgs, out, use_shift=false, parallel=false))]
fn forward_fp32_nchw(
    py: Python<'_>,
    imgs: PyReadonlyArrayDyn<'_, u8>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    use_shift: bool,
    parallel: bool,
) -> PyResult<()> {
    let in_slice = imgs.as_slice()?;
    let out_slice = out.as_slice_mut()?;
    let b = in_slice.len() / IMG_PIXELS;
    assert_eq!(in_slice.len(), b * IMG_PIXELS);
    assert_eq!(out_slice.len(), b * OUT_ELEMS);

    let in_addr = in_slice.as_ptr() as usize;
    let out_addr = out_slice.as_mut_ptr() as usize;
    let lut_addr = if use_shift {
        LUT_FP32_SHIFT.0.as_ptr() as usize
    } else {
        LUT_FP32_GRID.0.as_ptr() as usize
    };

    py.allow_threads(move || unsafe {
        if parallel && b > 1 {
            (0..b).into_par_iter().for_each(|i| {
                let lut_f32 = lut_addr as *const f32;
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut f32).add(i * OUT_ELEMS);
                kernel_fp32_nchw_single(img_p, out_p, lut_f32, use_shift);
            });
        } else {
            let lut_f32 = lut_addr as *const f32;
            for i in 0..b {
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut f32).add(i * OUT_ELEMS);
                kernel_fp32_nchw_single(img_p, out_p, lut_f32, use_shift);
            }
        }
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (imgs, out, parallel=false))]
fn forward_interp4d_nchw(
    py: Python<'_>,
    imgs: PyReadonlyArrayDyn<'_, u8>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    parallel: bool,
) -> PyResult<()> {
    let in_slice = imgs.as_slice()?;
    let out_slice = out.as_slice_mut()?;
    let b = in_slice.len() / IMG_PIXELS;
    assert_eq!(in_slice.len(), b * IMG_PIXELS);
    assert_eq!(out_slice.len(), b * OUT_ELEMS);

    let in_addr = in_slice.as_ptr() as usize;
    let out_addr = out_slice.as_mut_ptr() as usize;
    let lut_addr = LUT_FP32_GRID.0.as_ptr() as usize;

    py.allow_threads(move || unsafe {
        if parallel && b > 1 {
            (0..b).into_par_iter().for_each(|i| {
                let lut_f32 = lut_addr as *const f32;
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut f32).add(i * OUT_ELEMS);
                kernel_interp4d_nchw_single(img_p, out_p, lut_f32);
            });
        } else {
            let lut_f32 = lut_addr as *const f32;
            for i in 0..b {
                let img_p = (in_addr as *const u8).add(i * IMG_PIXELS);
                let out_p = (out_addr as *mut f32).add(i * OUT_ELEMS);
                kernel_interp4d_nchw_single(img_p, out_p, lut_f32);
            }
        }
    });
    Ok(())
}

// ============================================================================
// DIRECT RAW POINTER API (cho PyTorch Tensor .data_ptr() - < 80ns FFI overhead)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (in_ptr, out_ptr, batch_size, mode=0, use_shift=true, parallel=false))]
fn forward_ptr(
    in_ptr: usize,
    out_ptr: usize,
    batch_size: usize,
    mode: u8,
    use_shift: bool,
    parallel: bool,
) {
    // mode:
    //   0 -> INT8 NHWC (14x14x4)
    //   1 -> INT8 NCHW (4x14x14)
    //   2 -> FP32 NCHW (4x14x14)
    //   3 -> INTERP4D FP32 NCHW (4x14x14)
    unsafe {
        match mode {
            0 => {
                let lut_addr = if use_shift {
                    LUT_INT8_SHIFT.0.as_ptr() as usize
                } else {
                    LUT_INT8_GRID.0.as_ptr() as usize
                };
                if parallel && batch_size > 1 {
                    (0..batch_size).into_par_iter().for_each(|i| {
                        let lut_u32 = lut_addr as *const u32;
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut u32).add(i * NUM_PATCHES);
                        kernel_int8_nhwc_single(img_p, out_p, lut_u32, use_shift);
                    });
                } else {
                    let lut_u32 = lut_addr as *const u32;
                    for i in 0..batch_size {
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut u32).add(i * NUM_PATCHES);
                        kernel_int8_nhwc_single(img_p, out_p, lut_u32, use_shift);
                    }
                }
            }
            1 => {
                let lut_addr = if use_shift {
                    LUT_INT8_SHIFT.0.as_ptr() as usize
                } else {
                    LUT_INT8_GRID.0.as_ptr() as usize
                };
                if parallel && batch_size > 1 {
                    (0..batch_size).into_par_iter().for_each(|i| {
                        let lut_u32 = lut_addr as *const u32;
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut u8).add(i * OUT_ELEMS);
                        kernel_int8_nchw_single(img_p, out_p, lut_u32, use_shift);
                    });
                } else {
                    let lut_u32 = lut_addr as *const u32;
                    for i in 0..batch_size {
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut u8).add(i * OUT_ELEMS);
                        kernel_int8_nchw_single(img_p, out_p, lut_u32, use_shift);
                    }
                }
            }
            2 => {
                let lut_addr = if use_shift {
                    LUT_FP32_SHIFT.0.as_ptr() as usize
                } else {
                    LUT_FP32_GRID.0.as_ptr() as usize
                };
                if parallel && batch_size > 1 {
                    (0..batch_size).into_par_iter().for_each(|i| {
                        let lut_f32 = lut_addr as *const f32;
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut f32).add(i * OUT_ELEMS);
                        kernel_fp32_nchw_single(img_p, out_p, lut_f32, use_shift);
                    });
                } else {
                    let lut_f32 = lut_addr as *const f32;
                    for i in 0..batch_size {
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut f32).add(i * OUT_ELEMS);
                        kernel_fp32_nchw_single(img_p, out_p, lut_f32, use_shift);
                    }
                }
            }
            _ => {
                let lut_addr = LUT_FP32_GRID.0.as_ptr() as usize;
                if parallel && batch_size > 1 {
                    (0..batch_size).into_par_iter().for_each(|i| {
                        let lut_f32 = lut_addr as *const f32;
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut f32).add(i * OUT_ELEMS);
                        kernel_interp4d_nchw_single(img_p, out_p, lut_f32);
                    });
                } else {
                    let lut_f32 = lut_addr as *const f32;
                    for i in 0..batch_size {
                        let img_p = (in_ptr as *const u8).add(i * IMG_PIXELS);
                        let out_p = (out_ptr as *mut f32).add(i * OUT_ELEMS);
                        kernel_interp4d_nchw_single(img_p, out_p, lut_f32);
                    }
                }
            }
        }
    }
}

#[pymodule]
fn rust_qlut(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_int8_nhwc, m)?)?;
    m.add_function(wrap_pyfunction!(forward_int8_nchw, m)?)?;
    m.add_function(wrap_pyfunction!(forward_fp32_nchw, m)?)?;
    m.add_function(wrap_pyfunction!(forward_interp4d_nchw, m)?)?;
    m.add_function(wrap_pyfunction!(forward_ptr, m)?)?;
    Ok(())
}
