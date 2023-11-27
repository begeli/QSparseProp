#ifndef QSPARSEPROP_QSPARSEPROP_LINEAR_WRAPPER_H
#define QSPARSEPROP_QSPARSEPROP_LINEAR_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/sparse/qsparse_linear.h"

namespace py = pybind11;

namespace sparse {
    void sparse_linear_vectorized_forward_wrapper(
        py::array_t<int8_t> X, float X_scale,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val, float W_val_scale,
        py::array_t<float> O
    );

    void sparse_linear_vectorized_backward_wrapper(
        py::array_t<int8_t> X, float X_scale,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<int8_t> W_val, float W_val_scale, py::array_t<int8_t> dLdO, float dLdO_scale,
        py::array_t<float> dLdX, py::array_t<float> dLdW_val
    );

    void sparse_linear_vectorized_grouped_forward_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val,
        py::array_t<float> W_val_scale, int W_qgroup_size, py::array_t<float> O
    );

    void sparse_linear_vectorized_grouped_backward_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<int8_t> W_val, py::array_t<float> W_val_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale,  int dLdO_qgroup_size,
        py::array_t<float> dLdX, py::array_t<float> dLdW_val
    );

    void sparse_linear_vectorized_parallel_forward_wrapper(
        py::array_t<int8_t> X, float X_scale,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val, float W_val_scale,
        py::array_t<float> O
    );

    void sparse_linear_vectorized_parallel_backward_wrapper(
        py::array_t<int8_t> X, float X_scale,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<int8_t> W_val, float W_val_scale, py::array_t<int8_t> dLdO, float dLdO_scale,
        py::array_t<float> dLdX, py::array_t<float> dLdW_val
    );

    void sparse_linear_vectorized_grouped_parallel_forward_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M, py::array_t<int8_t> W_val,
        py::array_t<float> W_val_scale, int W_qgroup_size, py::array_t<float> O
    );

    void sparse_linear_vectorized_grouped_parallel_backward_wrapper(
        py::array_t<int8_t> X, py::array_t<float> X_scale, int X_qgroup_size,
        py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<int8_t> W_val, py::array_t<float> W_val_scale, int W_qgroup_size,
        py::array_t<int8_t> dLdO, py::array_t<float> dLdO_scale, int dLdO_qgroup_size,
        py::array_t<float> dLdX, py::array_t<float> dLdW_val
    );
}

#endif //QSPARSEPROP_QSPARSEPROP_LINEAR_WRAPPER_H
