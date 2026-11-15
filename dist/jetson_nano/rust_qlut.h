#ifndef RUST_QLUT_H
#define RUST_QLUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Returns size in bytes of the 4-bit INT8 Q-LUT table (262,144 bytes = 256 KiB). */
size_t qlut_table_int8_bytes(void);

/** Returns size in bytes of the 4-bit FP32 Q-LUT table (1,048,576 bytes = 1,024 KiB). */
size_t qlut_table_fp32_bytes(void);

/** Batch 28x28 -> 14x14x4 INT8 NHWC (u32 packed) Q-LUT forward pass. */
void qlut_forward_int8_nhwc_28x28(
    const uint8_t *img_ptr,
    uint32_t *out_u32_ptr,
    size_t batch_size,
    bool use_shift
);

/** Batch 28x28 -> 4x14x14 INT8 NCHW Q-LUT forward pass (ARMv8 NEON / x86_64 SSSE3). */
void qlut_forward_int8_nchw_28x28(
    const uint8_t *img_ptr,
    int8_t *out_i8_ptr,
    size_t batch_size,
    bool use_shift
);

/** Batch 28x28 -> 4x14x14 FP32 NCHW Interp4D Q-LUT forward pass (ARMv8 NEON FMA / AVX2). */
void qlut_forward_interp4d_nchw_28x28(
    const uint8_t *img_ptr,
    float *out_f32_ptr,
    size_t batch_size
);

/** Strided arbitrary-resolution INT8 NHWC Q-LUT forward pass. */
void qlut_forward_int8_nhwc_strided_2d(
    const uint8_t *img_ptr,
    uint32_t *out_u32_ptr,
    size_t img_h,
    size_t img_w,
    size_t stride,
    bool use_shift
);

/** Strided arbitrary-resolution FP32 NCHW Interp4D Q-LUT forward pass. */
void qlut_forward_interp4d_nchw_strided_2d(
    const uint8_t *img_ptr,
    float *out_f32_ptr,
    size_t img_h,
    size_t img_w,
    size_t stride
);

#ifdef __cplusplus
}
#endif

#endif /* RUST_QLUT_H */
