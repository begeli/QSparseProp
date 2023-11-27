#include "qsparseprop_linear_wrapper.h"

void sparse::sparse_linear_vectorized_forward_wrapper(
    py::array_t<int8_t> X, float X_scale,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val, float W_val_scale,
    py::array_t<float> O
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = O.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_sparse_linear_vectorized_forward(
        B, M, N, ptr_X, X_scale, ptr_W_idx_N, ptr_W_idx_M, ptr_W_val, W_val_scale, ptr_O
    );
}

void sparse::sparse_linear_vectorized_parallel_forward_wrapper(
    py::array_t<int8_t> X, float X_scale,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val, float W_val_scale,
    py::array_t<float> O
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = O.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_sparse_linear_vectorized_parallel_forward(
        B, M, N, ptr_X, X_scale, ptr_W_idx_N, ptr_W_idx_M, ptr_W_val, W_val_scale, ptr_O
    );
}

void sparse::sparse_linear_vectorized_backward_wrapper(
    py::array_t<int8_t> X, float X_scale,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<int8_t> W_val, float W_val_scale, py::array_t<int8_t> dLdO, float dLdO_scale,
    py::array_t<float> dLdX, py::array_t<float> dLdW_val
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = dLdO.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_linear_vectorized_backward(
        B, M, N, ptr_X, X_scale, ptr_W_idx_N, ptr_W_idx_M, ptr_W_val, W_val_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_linear_vectorized_parallel_backward_wrapper(
    py::array_t<int8_t> X, float X_scale,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<int8_t> W_val, float W_val_scale, py::array_t<int8_t> dLdO, float dLdO_scale,
    py::array_t<float> dLdX, py::array_t<float> dLdW_val
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = dLdO.shape()[0];

    auto buf_X = X.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_linear_vectorized_parallel_backward(
        B, M, N, ptr_X, X_scale, ptr_W_idx_N, ptr_W_idx_M, ptr_W_val, W_val_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_linear_vectorized_grouped_forward_wrapper(
    py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val,
    py::array_t<float> W_val_scale, int W_qgroup_size, py::array_t<float> O
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = O.shape()[0];

    auto buf_X = X.request();
    auto buf_X_scale = X_scale.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_W_val_scale = W_val_scale.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    float* ptr_X_scale = (float*) buf_X_scale.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_W_val_scale = (float*) buf_W_val_scale.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_grouped_sparse_linear_vectorized_forward(
        B, M, N, ptr_X, ptr_X_scale, X_qgroup_size, ptr_W_idx_N,
        ptr_W_idx_M, ptr_W_val, ptr_W_val_scale, W_qgroup_size, ptr_O
    );
}

void sparse::sparse_linear_vectorized_grouped_backward_wrapper(
    py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<int8_t> W_val, py::array_t<float> W_val_scale, int W_qgroup_size,
    py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale,  int dLdO_qgroup_size,
    py::array_t<float> dLdX, py::array_t<float> dLdW_val
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = dLdO.shape()[0];

    auto buf_X = X.request();
    auto buf_X_scale = X_scale.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_W_val_scale = W_val_scale.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdO_scale = dLdO_scale.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    float* ptr_X_scale = (float*) buf_X_scale.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_W_val_scale = (float*) buf_W_val_scale.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdO_scale = (float*) buf_dLdO_scale.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_grouped_sparse_linear_vectorized_backward(
        B, M, N, ptr_X, ptr_X_scale, X_qgroup_size, ptr_W_idx_N,
        ptr_W_idx_M, ptr_W_val, ptr_W_val_scale, W_qgroup_size, ptr_dLdO,
        ptr_dLdO_scale, dLdO_qgroup_size, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_linear_vectorized_grouped_parallel_forward_wrapper(
    py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val,
    py::array_t<float> W_val_scale, int W_qgroup_size, py::array_t<float> O
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = O.shape()[0];

    auto buf_X = X.request();
    auto buf_X_scale = X_scale.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_W_val_scale = W_val_scale.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    float* ptr_X_scale = (float*) buf_X_scale.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_W_val_scale = (float*) buf_W_val_scale.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_grouped_sparse_linear_vectorized_parallel_forward(
        B, M, N, ptr_X, ptr_X_scale, X_qgroup_size, ptr_W_idx_N,
        ptr_W_idx_M, ptr_W_val, ptr_W_val_scale, W_qgroup_size, ptr_O
    );
}

void sparse::sparse_linear_vectorized_grouped_parallel_backward_wrapper(
    py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
    py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
    py::array_t<int8_t> W_val, py::array_t<float> W_val_scale, int W_qgroup_size,
    py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size,
    py::array_t<float> dLdX, py::array_t<float> dLdW_val
) {
    int B = X.shape()[1];
    int M = X.shape()[0];
    int N = dLdO.shape()[0];

    auto buf_X = X.request();
    auto buf_X_scale = X_scale.request();
    auto buf_W_idx_N = W_idx_N.request();
    auto buf_W_idx_M = W_idx_M.request();
    auto buf_W_val = W_val.request();
    auto buf_W_val_scale = W_val_scale.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdO_scale = dLdO_scale.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    float* ptr_X_scale = (float*) buf_X_scale.ptr;
    int* ptr_W_idx_N = (int*) buf_W_idx_N.ptr;
    int* ptr_W_idx_M = (int*) buf_W_idx_M.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_W_val_scale = (float*) buf_W_val_scale.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdO_scale = (float*) buf_dLdO_scale.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_grouped_sparse_linear_vectorized_parallel_backward(
        B, M, N, ptr_X, ptr_X_scale, X_qgroup_size, ptr_W_idx_N,
        ptr_W_idx_M, ptr_W_val, ptr_W_val_scale, W_qgroup_size, ptr_dLdO,
        ptr_dLdO_scale, dLdO_qgroup_size, ptr_dLdX, ptr_dLdW_val
    );
}

