#ifndef QSPARSEPROP_TENSOR_UTILS_WRAPPER_H
#define QSPARSEPROP_TENSOR_UTILS_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/utils/tensor_utils.h"

namespace py = pybind11;

void transpose_wrapper(py::array_t<float> X, py::array_t<float> XT, int block_size);

#endif //QSPARSEPROP_TENSOR_UTILS_WRAPPER_H
