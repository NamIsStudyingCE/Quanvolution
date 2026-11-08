#ifndef RUST_QLUT_H
#define RUST_QLUT_H

/*
 * rust_qlut.h — C/C++ FFI Header for Precompiled Quantum Convolution Lookup Table (Q-LUT)
 * Target Hardware:
 *   - x86_64 (AVX2 / FMA / SSSE3)
 *   - NVIDIA Jetson Nano (Quad-core ARM Cortex-A57 @ 1.43 GHz, ARMv8-A NEON SIMD, 2 MB L2 Cache)
 *
 * Author: Nguyen Hoang Nam (MSSV: 22520916) - Faculty of Computer Engineering, UIT VNU-HCM
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns the exact byte footprint of the 4-bit INT8 Q-LUT table (262,144 bytes = 256 KiB).
 * Fits entirely inside the 2 MB L2 Cache of NVIDIA Jetson Nano (ARM Cortex-A57).
 */
size_t qlut_table_int8_bytes(void);

/**
 * Returns the exact byte footprint of the 4-bit FP32 Q-LUT table (1,048,576 bytes = 1,024 KiB).
 */
size_t qlut_table_fp32_bytes(void);

/**
 * Fast 0-FLOP INT8 Quanvolution Lookup for 28x28 images (NHWC packed 4-channel output).
 *
 * @param img_ptr       Pointer to input uint8 images [batch_size, 28, 28]
 * @param out_u32_ptr   Pointer to output buffer [batch_size, 14, 14] of packed 4x int8 (as uint32_t)
 * @param batch_size    Number of images in batch
 * @param use_shift     If true, uses zero-arithmetic bit-shift index (p >> 4); if false, uses round grid map
 */
void qlut_forward_int8_nhwc_28x28(
    const uint8_t* img_ptr,
    uint32_t* out_u32_ptr,
    size_t batch_size,
    bool use_shift
);

/**
 * Fast 0-FLOP INT8 Quanvolution Lookup for 28x28 images (NCHW planar output).
 *
 * @param img_ptr       Pointer to input uint8 images [batch_size, 28, 28]
 * @param out_i8_ptr    Pointer to output int8 tensor [batch_size, 4, 14, 14]
 * @param batch_size    Number of images in batch
 * @param use_shift     If true, uses zero-arithmetic bit-shift index (p >> 4)
 */
void qlut_forward_int8_nchw_28x28(
    const uint8_t* img_ptr,
    int8_t* out_i8_ptr,
    size_t batch_size,
    bool use_shift
);

/**
 * Continuous 4D Multilinear SIMD Interpolation (AVX2 / ARMv8 NEON vfmaq_f32) for 28x28 images.
 *
 * @param img_ptr       Pointer to input uint8 images [batch_size, 28, 28]
 * @param out_f32_ptr   Pointer to output float32 tensor [batch_size, 4, 14, 14]
 * @param batch_size    Number of images in batch
 */
void qlut_forward_interp4d_nchw_28x28(
    const uint8_t* img_ptr,
    float* out_f32_ptr,
    size_t batch_size
);

/**
 * Arbitrary-Resolution 2D Strided INT8 Q-LUT Lookup (from 28x28 up to 1024x1024 Full-HD X-Ray).
 * Memory complexity is O(1) auxiliary L2 cache (constant 256 KiB table).
 *
 * @param img_ptr       Pointer to input uint8 image [img_h, img_w]
 * @param out_u32_ptr   Pointer to output packed int8x4 buffer [out_h, out_w], where out_h = (img_h - 1) / stride
 * @param img_h         Input image height
 * @param img_w         Input image width
 * @param stride        Spatial convolution stride (e.g., 2 or 4)
 * @param use_shift     If true, uses bit-shift index (p >> 4)
 */
void qlut_forward_int8_nhwc_strided_2d(
    const uint8_t* img_ptr,
    uint32_t* out_u32_ptr,
    size_t img_h,
    size_t img_w,
    size_t stride,
    bool use_shift
);

/**
 * Arbitrary-Resolution 2D Strided 4D SIMD Interpolation (AVX2 / ARMv8 NEON).
 *
 * @param img_ptr       Pointer to input uint8 image [img_h, img_w]
 * @param out_f32_ptr   Pointer to output float32 planar buffer [4, out_h, out_w]
 * @param img_h         Input image height
 * @param img_w         Input image width
 * @param stride        Spatial convolution stride (e.g., 2 or 4)
 */
void qlut_forward_interp4d_nchw_strided_2d(
    const uint8_t* img_ptr,
    float* out_f32_ptr,
    size_t img_h,
    size_t img_w,
    size_t stride
);

#ifdef __cplusplus
}
#endif

#endif /* RUST_QLUT_H */
