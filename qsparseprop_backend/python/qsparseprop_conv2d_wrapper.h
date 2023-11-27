#ifndef QSPARSEPROP_QSPARSEPROP_CONV2D_WRAPPER_H
#define QSPARSEPROP_QSPARSEPROP_CONV2D_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/sparse/qsparse_conv2d_stride_1.h"
#include "src/sparse/qsparse_conv2d_stride_2.h"
#include "src/utils/sparse_utils.h"

namespace py = pybind11;

namespace sparse {
    void sparse_conv2d_vectorized_forward_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_parallel_forward_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_forward_stride_1_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_parallel_forward_stride_1_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<float> O, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_backward_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_parallel_backward_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_backward_stride_1_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_parallel_backward_stride_1_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_forward_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_parallel_forward_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_forward_stride_2_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<float> O, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_parallel_forward_stride_2_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<float> O, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_backward_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_parallel_backward_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_backward_stride_2_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
    void sparse_conv2d_vectorized_grouped_parallel_backward_stride_2_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, py::array_t<float> W_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void sparsify_conv2d_wrapper(
        int OC, int IC, int K, py::array_t<float> W, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y,py::array_t<float> W_val
    );

    void densify_conv2d_wrapper(
        int OC, int IC, int K, py::array_t<float> W, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y,py::array_t<float> W_val
    );
}

#endif //QSPARSEPROP_QSPARSEPROP_CONV2D_WRAPPER_H
