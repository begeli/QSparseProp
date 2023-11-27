#include "sparseprop_linear_wrapper.h"

void sparse::uq_sparse_linear_vectorized_forward_wrapper(
    py::array_t<float> X, py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<float> W_val, py::array_t<float> O
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = O.shape()[0];
    int W_nnz = W_val.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    float* ptr_X = (float*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::sparse_linear_vectorized_forward(
        B, M, N, W_nnz, ptr_X, ptr_W_idx_N, ptr_W_idx_M, ptr_W_val, ptr_O
    );
}

void sparse::uq_sparse_linear_vectorized_backward_wrapper(
    py::array_t<float> X, py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<float> W_val, py::array_t<float> dLdO, py::array_t<float> dLdX,
    py::array_t<float> dLdW_val
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = dLdO.shape()[0];
    int W_nnz = W_val.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    float* ptr_X = (float*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    float* ptr_W_val = (float*) buf_W_val.ptr;
    float* ptr_dLdO = (float*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::sparse_linear_vectorized_backward(
        B, M, N, W_nnz, ptr_X, ptr_W_idx_N, ptr_W_idx_M,
        ptr_W_val, ptr_dLdO, ptr_dLdX,ptr_dLdW_val
    );
}
