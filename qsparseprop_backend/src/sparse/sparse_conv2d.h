#ifndef QSPARSEPROP_SPARSE_CONV2D_H
#define QSPARSEPROP_SPARSE_CONV2D_H

#include <immintrin.h>
#include <algorithm>

namespace sparse {
    void sparse_conv2d_vectorized_forward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, float* W_val, float* O
    );
    void sparse_conv2d_vectorized_backward_stride_1(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, float* W_val, float* dLdO, float* dLdX, float* dLdW_val
    );

    void sparse_conv2d_vectorized_forward_stride_2(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, float* W_val, float* O
    );
    void sparse_conv2d_vectorized_backward_stride_2(
        int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
        float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, float* W_val, float* dLdO, float* dLdX, float* dLdW_val
    );
}

#endif //QSPARSEPROP_SPARSE_CONV2D_H
