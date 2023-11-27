#ifndef QSPARSEPROP_SPARSE_LINEAR_H
#define QSPARSEPROP_SPARSE_LINEAR_H

#include <immintrin.h>

namespace sparse {
    void sparse_linear_vectorized_forward(
        int B, int M, int N, int W_nnz,
        float* X, int* W_idx_N, int* W_idx_M,
        float* W_val, float* O
    );

    void sparse_linear_vectorized_backward(
        int B, int M, int N, int W_nnz, float* X,
        int* W_idx_N, int* W_idx_M,float* W_val,
        float* dLdO, float* dLdX, float* dLdW_val
    );
}

#endif //QSPARSEPROP_SPARSE_LINEAR_H
