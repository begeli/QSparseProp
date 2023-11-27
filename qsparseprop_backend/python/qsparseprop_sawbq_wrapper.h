#ifndef QSPARSEPROP_QSPARSEPROP_SAWBQ_WRAPPER_H
#define QSPARSEPROP_QSPARSEPROP_SAWBQ_WRAPPER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/quantization/sawb_quantization8_strategy.h"

namespace py = pybind11;

namespace quantization {
    float sawb_8_bit_scalar_quantization_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        float lowerBound,
        float upperBound,
        float weight1,
        float weight2
    );

    float sawb_8_bit_quantization_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        float lowerBound,
        float upperBound,
        float weight1,
        float weight2
    );

    float sawb_8_bit_quantization_parallel_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        float lowerBound,
        float upperBound,
        float weight1,
        float weight2
    );

    void sawb_8_bit_quantization_grouped_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        py::array_t<float> scales,
        py::array_t<float> dq_consts,
        int group_size,
        int group_shift_amount,
        float lowerBound,
        float upperBound,
        float weight1,
        float weight2
    );

    void sawb_8_bit_quantization_grouped_parallel_wrapper(
        py::array_t<float> dq_values,
        py::array_t<int8_t> q_values,
        py::array_t<float> scales,
        py::array_t<float> dq_consts,
        int group_size,
        int group_shift_amount,
        float lowerBound,
        float upperBound,
        float weight1,
        float weight2
    );
}

#endif //QSPARSEPROP_QSPARSEPROP_SAWBQ_WRAPPER_H
