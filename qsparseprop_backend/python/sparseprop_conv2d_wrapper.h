#ifndef QSPARSEPROP_SPARSEPROP_CONV2D_WRAPPER_H
#define QSPARSEPROP_SPARSEPROP_CONV2D_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/sparse/sparse_conv2d.h"

namespace py = pybind11;

namespace sparse {
    void uq_sparse_conv2d_vectorized_forward_stride_1_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
        py::array_t<float> O, int kernel_size, int padding
    );
    void uq_sparse_conv2d_vectorized_backward_stride_1_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
        py::array_t<float> dLdO, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );

    void uq_sparse_conv2d_vectorized_forward_stride_2_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
        py::array_t<float> O, int kernel_size, int padding
    );
    void uq_sparse_conv2d_vectorized_backward_stride_2_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_OC,
        py::array_t<int16_t> W_idx_IC, py::array_t<uint8_t> W_idx_X,
        py::array_t<uint8_t> W_idx_Y, py::array_t<float> W_val,
        py::array_t<float> dLdO, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val, int kernel_size, int padding
    );
}

#endif //QSPARSEPROP_SPARSEPROP_CONV2D_WRAPPER_H
