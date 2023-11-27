#include "tensor_utils_wrapper.h"

void transpose_wrapper(py::array_t<float> X, py::array_t<float> XT, int block_size) {
    int N = X.shape()[0];
    int M = X.shape()[1];

    auto buf_X = X.request();
    auto buf_XT = XT.request();

    float* ptr_X = (float*) buf_X.ptr;
    float* ptr_XT = (float*) buf_XT.ptr;

    transpose(ptr_X, ptr_XT, N, M, block_size);
}
