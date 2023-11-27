#include "qsparseprop_conv2d_over_on_wrapper.h"

void sparse::sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<float> O, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = O.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_sparse_conv2d_vectorized_forward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding,
        ptr_X, X_scale, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_O
    );
}

void sparse::sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
    py::array_t<float> dLdW_val, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, ptr_X, X_scale,
        ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
    py::array_t<float> dLdW_val, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_2(
        B, IC, OC, M, N, K, W_nnz, padding, ptr_X, X_scale,
        ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_conv2d_vectorized_parallel_forward_over_on_stride_1_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<float> O, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = O.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_O = O.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    float* ptr_O = (float*) buf_O.ptr;

    sparse::quantized_sparse_conv2d_vectorized_parallel_forward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding,
        ptr_X, X_scale, ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_O
    );
}

void sparse::sparse_conv2d_vectorized_parallel_backward_over_on_stride_1_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
    py::array_t<float> dLdW_val, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, ptr_X, X_scale,
        ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}

void sparse::sparse_conv2d_vectorized_parallel_backward_over_on_stride_2_wrapper(
    py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
    py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
    py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
    py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
    py::array_t<float> dLdW_val, int kernel_size, int padding
) {
    int B = X.shape()[0];
    int IC = X.shape()[1];
    int M = X.shape()[2];
    int N = X.shape()[3];
    int OC = dLdO.shape()[1];
    int W_nnz = W_val.shape()[0];
    int K = kernel_size;

    auto buf_X = X.request();
    auto buf_W_idx_OC = W_idx_OC.request();
    auto buf_W_idx_IC = W_idx_IC.request();
    auto buf_W_idx_X = W_idx_X.request();
    auto buf_W_idx_Y = W_idx_Y.request();
    auto buf_W_val = W_val.request();
    auto buf_dLdO = dLdO.request();
    auto buf_dLdX = dLdX.request();
    auto buf_dLdW_val = dLdW_val.request();

    int8_t* ptr_X = (int8_t*) buf_X.ptr;
    int* ptr_W_idx_OC = (int*) buf_W_idx_OC.ptr;
    int16_t* ptr_W_idx_IC = (int16_t*) buf_W_idx_IC.ptr;
    uint8_t* ptr_W_idx_X = (uint8_t*) buf_W_idx_X.ptr;
    uint8_t* ptr_W_idx_Y = (uint8_t*) buf_W_idx_Y.ptr;
    int8_t* ptr_W_val = (int8_t*) buf_W_val.ptr;
    int8_t* ptr_dLdO = (int8_t*) buf_dLdO.ptr;
    float* ptr_dLdX = (float*) buf_dLdX.ptr;
    float* ptr_dLdW_val = (float*) buf_dLdW_val.ptr;

    sparse::quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_2(
        B, IC, OC, M, N, K, W_nnz, padding, ptr_X, X_scale,
        ptr_W_idx_OC, ptr_W_idx_IC, ptr_W_idx_X, ptr_W_idx_Y,
        ptr_W_val, W_scale, ptr_dLdO, dLdO_scale, ptr_dLdX, ptr_dLdW_val
    );
}