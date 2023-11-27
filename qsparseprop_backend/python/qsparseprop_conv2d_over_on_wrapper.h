#ifndef QSPARSEPROP_QSPARSEPROP_CONV2D_OVER_ON_H
#define QSPARSEPROP_QSPARSEPROP_CONV2D_OVER_ON_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/sparse/qsparse_conv2d_over_on.h"

namespace py = pybind11;

namespace sparse {
    void sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_parallel_forward_over_on_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<float> O, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_parallel_backward_over_on_stride_1_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void sparse_conv2d_vectorized_parallel_backward_over_on_stride_2_wrapper(
        py::array_t<int8_t> X, float X_scale, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<int8_t> W_val, float W_scale,
        py::array_t<int8_t> dLdO, float dLdO_scale, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
}

#endif //QSPARSEPROP_QSPARSEPROP_CONV2D_OVER_ON_H
