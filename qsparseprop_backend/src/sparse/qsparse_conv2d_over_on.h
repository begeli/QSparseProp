#ifndef QSPARSEPROP_QSPARSE_CONV2D_OVER_ON_H
#define QSPARSEPROP_QSPARSE_CONV2D_OVER_ON_H

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "src/quantization/quantization8_strategy.h"
#include "src/utils/simdmath.h"

namespace sparse {
    void quantized_sparse_conv2d_vectorized_forward_over_on_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
    );

    void quantized_sparse_conv2d_vectorized_backward_over_on_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
        float* dLdX, float* dLdW_val
    );

    void quantized_sparse_conv2d_vectorized_backward_over_on_stride_2(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
        float* dLdX, float* dLdW_val
    );

    void quantized_sparse_conv2d_vectorized_parallel_forward_over_on_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
    );

    void quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
        float* dLdX, float* dLdW_val
    );

    void quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_2(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
        float* dLdX, float* dLdW_val
    );
}

#endif //QSPARSEPROP_QSPARSE_CONV2D_OVER_ON_H
