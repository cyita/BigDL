#include "llamacpp/ggml.h"
// #include "llamacpp/ggml-quants.h"
#include<iostream>
#include <chrono>

using namespace std::chrono;

// // FP16 <-> FP32
// // ref: https://github.com/Maratyszcza/FP16

// static inline float fp32_from_bits(uint32_t w) {
//     union {
//         uint32_t as_bits;
//         float as_value;
//     } fp32;
//     fp32.as_bits = w;
//     return fp32.as_value;
// }

// static inline uint32_t fp32_to_bits(float f) {
//     union {
//         float as_value;
//         uint32_t as_bits;
//     } fp32;
//     fp32.as_value = f;
//     return fp32.as_bits;
// }

// static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
// #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
//     const float scale_to_inf = 0x1.0p+112f;
//     const float scale_to_zero = 0x1.0p-110f;
// #else
//     const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
//     const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
// #endif
//     float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

//     const uint32_t w = fp32_to_bits(f);
//     const uint32_t shl1_w = w + w;
//     const uint32_t sign = w & UINT32_C(0x80000000);
//     uint32_t bias = shl1_w & UINT32_C(0xFF000000);
//     if (bias < UINT32_C(0x71000000)) {
//         bias = UINT32_C(0x71000000);
//     }

//     base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
//     const uint32_t bits = fp32_to_bits(base);
//     const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
//     const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
//     const uint32_t nonsign = exp_bits + mantissa_bits;
//     return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
// }

// #define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

// #if !defined(GGML_FP32_TO_FP16)
// #define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
// #endif

// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MAX(a, b) ((a) > (b) ? (a) : (b))

void ggml_requantize_to_npu_q4_0(int m, int n, float* y) {
    int64_t npu_qk = 2;        // 2
    size_t npu_block_size_in_bytes = 1;    // 1
    size_t npu_dst_size = (n * m / npu_qk) * npu_block_size_in_bytes + sizeof(ggml_fp16_t) * m;
    uint8_t *q4_0_dst = (uint8_t *)malloc(npu_dst_size * sizeof(uint8_t));
    ggml_fp16_t *q4_0_scale = (ggml_fp16_t *)((uint8_t*)q4_0_dst + (m * n /2));
    ggml_quantize_chunk(GGML_TYPE_Q4_0_RTN, y, q4_0_dst, 0, m, n, NULL);
    free(q4_0_dst);
    q4_0_dst = NULL;
}

void ggml_requantize_to_npu_q4_1(int m, int n, float* y) {
    const int dst_size = m * (n/2 + 2 * 2); // byte size
    uint8_t* dst = (uint8_t*)malloc(dst_size);
    ggml_quantize_chunk(GGML_TYPE_Q4_1_RTN, y, dst, 0, m, n, NULL);
    ggml_fp16_t* scale = (ggml_fp16_t*)((uint8_t*)dst + (m * n /2));
    ggml_fp16_t* zero = scale + m;
    free(dst);
    dst = NULL;
}

// static void quantize_row_q4_1_rtn_impl(const float * x, uint8_t * y, ggml_fp16_t * scale_y, ggml_fp16_t * min_y, int b, int k) {
//     float min = FLT_MAX;
//     float max = -FLT_MAX;

//     for (int j = 0; j < k; j++) {
//         const float v = x[j];

//         if (v < min) min = v;
//         if (v > max) max = v;
//     }

//     const float d  = (max - min) / ((1 << 4) - 1);
//     const float id = d ? 1.0f/d : 0.0f;
//     scale_y[b/k] = GGML_FP32_TO_FP16(d * sqrt(k));
//     min_y[b/k] = GGML_FP32_TO_FP16((min + 8.0 * d) * sqrt(k));

//     for (int j = 0; j < k/2; ++j) {

//         float x0 = (x[2*j] - min)*id;
//         float x1 = (x[2*j + 1] - min)*id;

//         // v1: directly use uint4_t for asym_int4
//         // const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
//         // const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));
//         // v2: use int4_t for asym_int4
//         const uint8_t xi0_ = MIN(15, (int8_t)(x0 + 0.5f));
//         const uint8_t xi1_ = MIN(15, (int8_t)(x1 + 0.5f));

//         const int8_t x10 = xi0_ - 8;
//         const int8_t x20 = xi1_ - 8;

//         const uint8_t xi0 = x10 & 0x0F;
//         const uint8_t xi1 = x20 & 0x0F;

//         y[j]  = xi0;
//         y[j] |= xi1 << 4;
//     }
// }

// size_t quantize_q4_1_rtn2(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
//     const int row_size = n_per_row / 2;
//     ggml_fp16_t *scale = (ggml_fp16_t *)((uint8_t *)dst + row_size * nrow);
//     ggml_fp16_t *zero = scale + nrow;
//     int64_t b = 0;
//     #pragma omp parallel for schedule(dynamic, 1)
//     for (b = 0; b < nrow * n_per_row ; b += n_per_row) {
//         uint8_t * y = (uint8_t *)dst + b / 2;
//         quantize_row_q4_1_rtn_impl(src + b, y, scale, zero, b, n_per_row);
//     }
//     return nrow * row_size;
// }

// static void quantize_row_q4_0_rtn_reference(const float * x, uint8_t * y, ggml_fp16_t * scale_y, int b, int k) {

//     float amax = 0.0f; // absolute max

//     for (int j = 0; j < k; j++) {
//         const float v = x[j];
//         if (amax < fabsf(v)) {
//             amax = fabsf(v);
//         }
//     }

//     const float d = amax / -8;
//     const float id = d ? 1.0f/d : 0.0f;


//     scale_y[b/k] = GGML_FP32_TO_FP16(d * sqrt(k));

//     for (int j = 0; j < k/2; ++j) {

//         // const int8_t x0 = MAX(-8, MIN(7, roundf(x[2*j]*id)));
//         // const int8_t x1 = MAX(-8, MIN(7, roundf(x[2*j+1]*id)));
//         const int8_t x0 = MAX(-8, MIN(7, x[2*j]*id));
//         const int8_t x1 = MAX(-8, MIN(7, x[2*j+1]*id));

//         const uint8_t xi0 = x0 & 0x0F;
//         const uint8_t xi1 = x1 & 0x0F;

//         y[j]  = xi0;
//         y[j] |= xi1 << 4;
//     }
// }

// size_t quantize_q4_0_rtn2(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
//     const int row_size = n_per_row / 2;
//     ggml_fp16_t *scale = (ggml_fp16_t *)((uint8_t *)dst + row_size * nrow);
//     int64_t b = 0;
//     // float scale_factor = sqrt(n_per_row);
//     #pragma omp parallel for schedule(dynamic, 1)
//     for (b = 0; b < nrow * n_per_row ; b += n_per_row) {
//         uint8_t * y = (uint8_t *)dst + b / 2;

//         quantize_row_q4_0_rtn_reference(src + b, y, scale, b, n_per_row);
//     }
//     return nrow * row_size;
// }

int main(int argc, char ** argv) {
    int m = 3072;
    int n = 3072;

    std::cout << "m: " << m << " n: " << n << std::endl;

    const size_t start_row = 0;
    const size_t row_size  = ggml_row_size(GGML_TYPE_Q4_0_RTN, n);

    float* y = (float*)malloc(m * n * sizeof(float));

    const int dst_size = m * (n/2 + 2 * 2); // byte size
    uint8_t* dst = (uint8_t*)malloc(dst_size);

    auto t1 = high_resolution_clock::now();

    
    for (int i=0; i < 1000; i ++) {
        ggml_quantize_chunk(GGML_TYPE_Q4_1_RTN, y, dst, 0, m, n, NULL);
        // quantize_q4_1_rtn2(y + 0, (char *) dst + start_row * row_size, m, n, NULL);
    }
    auto t2 = high_resolution_clock::now();

    ggml_fp16_t* scale = (ggml_fp16_t*)((uint8_t*)dst + (m * n /2));
    ggml_fp16_t* zero = scale + m;
    free(dst);
    dst = NULL;

    int64_t npu_qk = 2;        // 2
    size_t npu_block_size_in_bytes = 1;    // 1
    size_t npu_dst_size = (n * m / npu_qk) * npu_block_size_in_bytes + sizeof(ggml_fp16_t) * m;
    uint8_t *q4_0_dst = (uint8_t *)malloc(npu_dst_size * sizeof(uint8_t));
    ggml_fp16_t *q4_0_scale = (ggml_fp16_t *)((uint8_t*)q4_0_dst + (m * n /2));

    auto t3 = high_resolution_clock::now();
    for (int i=0; i < 1000; i ++) {
        ggml_quantize_chunk(GGML_TYPE_Q4_0_RTN, y, q4_0_dst, 0, m, n, NULL);
        // quantize_q4_0_rtn2(y + 0, (char *) q4_0_dst + start_row * row_size, m, n, NULL);
    }
    auto t4 = high_resolution_clock::now();

    free(q4_0_dst);
    q4_0_dst = NULL;
    
    free(y);
    y = NULL;
    std::cout << "Q4_1 avg time: " << (double)duration_cast<milliseconds>(t2 - t1).count()/1000 << " ms." << std::endl;
    std::cout << "Q4_0 avg time: " << (double)duration_cast<milliseconds>(t4 - t3).count()/1000 << " ms." << std::endl;
}