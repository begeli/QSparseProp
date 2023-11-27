#ifndef QSPARSEPROP_TENSOR_UTILS_H
#define QSPARSEPROP_TENSOR_UTILS_H

#include <immintrin.h>

void transpose(float* X, float* XT, const int N, const int M, const int block_size);

void transpose_16x16(float* mat, float* matT, const int lda, const int ldb);

#endif //QSPARSEPROP_TENSOR_UTILS_H
