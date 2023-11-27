#ifndef QSPARSEPROP_SPARSE_UTILS_H
#define QSPARSEPROP_SPARSE_UTILS_H

#include <cstdint>

void sparsify_conv2d(
    int IC, int OC, int K, float* W, int* W_idx_OC,
    int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val
);

void densify_conv2d(
    int IC, int OC, int K, float* W, int* W_idx_OC,
    int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val
);

#endif //QSPARSEPROP_SPARSE_UTILS_H
