#ifndef QSPARSEPROP_QSPARSEPROP_STDQ_WRAPPER_H
#define QSPARSEPROP_QSPARSEPROP_STDQ_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/quantization/standard_quantization8_strategy.h"

namespace py = pybind11;

namespace quantization {
    std::tuple<float, float> standard_8_bit_quantization_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        float lowerBound,
        float upperBound
    );

    std::tuple<float, float> standard_8_bit_quantization_parallel_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        float lowerBound,
        float upperBound
    );

    void standard_8_bit_quantization_grouped_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        py::array_t<float> scales,
        py::array_t<float> dq_consts,
        int group_size,
        int group_shift_amount,
        float lowerBound,
        float upperBound
    );

    void standard_8_bit_quantization_grouped_parallel_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        py::array_t<float> scales,
        py::array_t<float> dq_consts,
        int group_size,
        int group_shift_amount,
        float lowerBound,
        float upperBound
    );
}

#endif //QSPARSEPROP_QSPARSEPROP_STDQ_WRAPPER_H
