#ifndef QSPARSEPROP_QSPARSE_CONV2D_STRIDE_1_H
#define QSPARSEPROP_QSPARSE_CONV2D_STRIDE_1_H

#include <immintrin.h>
#include "src/quantization/quantization8_strategy.h"
#include <algorithm>

namespace sparse {
    void quantized_sparse_conv2d_vectorized_forward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
    );
    void quantized_sparse_conv2d_vectorized_parallel_forward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
    );
    void quantized_grouped_sparse_conv2d_vectorized_forward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
    );
    void quantized_grouped_sparse_conv2d_vectorized_parallel_forward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
    );

    void quantized_sparse_conv2d_vectorized_backward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale,
        int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
    );
    void quantized_sparse_conv2d_vectorized_parallel_backward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float W_scale,
        int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
    );
    void quantized_grouped_sparse_conv2d_vectorized_backward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size,
        int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
    );
    void quantized_grouped_sparse_conv2d_vectorized_parallel_backward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size,
        int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
    );
}

#endif //QSPARSEPROP_QSPARSE_CONV2D_STRIDE_1_H
