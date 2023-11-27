#ifndef QSPARSEPROP_QSPARSE_LINEAR_H
#define QSPARSEPROP_QSPARSE_LINEAR_H

#include "src/utils/simdmath.h"
#include "src/quantization/quantization8_strategy.h"
#include <immintrin.h>

// Currently, these do not support standard and logarithmic unbiased quantization

namespace sparse {
    // Requires: Dense input tensor, sparse weight tensor, output tensor
    void quantized_sparse_linear_vectorized_forward(
        int B, int M, int N, int8_t* X, float X_scale,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale, float* O
    );
    void quantized_sparse_linear_vectorized_parallel_forward(
        int B, int M, int N, int8_t* X, float X_scale,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale, float* O
    );
    void quantized_grouped_sparse_linear_vectorized_forward(
        int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
    );
    void quantized_grouped_sparse_linear_vectorized_parallel_forward(
        int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
    );

    // Requires: Dense input tensor, sparse weight tensor, Output gradient, Weight gradient, Input gradient
    void quantized_sparse_linear_vectorized_backward(
        int B, int M, int N, int8_t* X, float X_scale,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale,
        int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
    );
    void quantized_sparse_linear_vectorized_parallel_backward(
        int B, int M, int N, int8_t* X, float X_scale,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale,
        int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
    );
    void quantized_grouped_sparse_linear_vectorized_backward(
        int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size,
        int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
    );
    void quantized_grouped_sparse_linear_vectorized_parallel_backward(
        int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
        int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size,
        int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
    );
}

#endif //QSPARSEPROP_QSPARSE_LINEAR_H
