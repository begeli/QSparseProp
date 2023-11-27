#ifndef QSPARSEPROP_SPARSEPROP_LINEAR_WRAPPER_H
#define QSPARSEPROP_SPARSEPROP_LINEAR_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/sparse/sparse_linear.h"

namespace py = pybind11;

namespace sparse {
    void uq_sparse_linear_vectorized_forward_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<float> W_val, py::array_t<float> O
    );

    void uq_sparse_linear_vectorized_backward_wrapper(
        py::array_t<float> X, py::array_t<int> W_idx_N, py::array_t<int> W_idx_M,
        py::array_t<float> W_val, py::array_t<float> dLdO, py::array_t<float> dLdX,
        py::array_t<float> dLdW_val
    );
}

#endif //QSPARSEPROP_SPARSEPROP_LINEAR_WRAPPER_H
