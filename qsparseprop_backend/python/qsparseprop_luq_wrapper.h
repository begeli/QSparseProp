#ifndef QSPARSEPROP_QSPARSEPROP_LUQ_WRAPPER_H
#define QSPARSEPROP_QSPARSEPROP_LUQ_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/quantization/luq8_strategy.h"

namespace py = pybind11;

namespace quantization {
    float logarithmic_unbiased_8_bit_quantization_wrapper(py::array_t<float> dq_values, py::array_t<int8_t> q_values);
}

#endif //QSPARSEPROP_QSPARSEPROP_LUQ_WRAPPER_H
