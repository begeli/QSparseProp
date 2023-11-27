#include "qsparseprop_stdq_wrapper.h"

std::tuple<float, float> quantization::standard_8_bit_quantization_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    float lowerBound,
    float upperBound
) {
    int size = (int) dq_values.size();

    auto buf_dq_values = dq_values.request();
    auto buf_q_values = q_values.request();

    float* dq_values_ptr = (float*) buf_dq_values.ptr;
    int8_t* q_values_ptr = (int8_t*) buf_q_values.ptr;
    float scale;
    float dequantization_const;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_values_ptr, q_values_ptr, size, scale, dequantization_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(lowerBound, upperBound);
    strategy.quantize(input);

    return std::make_tuple(input.std_quantization_input.scale, input.std_quantization_input.dequantization_const);
}

std::tuple<float, float> quantization::standard_8_bit_quantization_parallel_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    float lowerBound,
    float upperBound
) {
    int size = (int) dq_values.size();

    auto buf_dq_values = dq_values.request();
    auto buf_q_values = q_values.request();

    float* dq_values_ptr = (float*) buf_dq_values.ptr;
    int8_t* q_values_ptr = (int8_t*) buf_q_values.ptr;
    float scale;
    float dequantization_const;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_values_ptr, q_values_ptr, size, scale, dequantization_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(lowerBound, upperBound);
    strategy.quantize_parallel(input);

    return std::make_tuple(input.std_quantization_input.scale, input.std_quantization_input.dequantization_const);
}

void quantization::standard_8_bit_quantization_grouped_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    py::array_t<float> scales,
    py::array_t<float> dq_consts,
    int group_size,
    int group_shift_amount,
    float lowerBound,
    float upperBound
) {
    int size = (int) dq_values.size();

    auto buf_dq_values = dq_values.request();
    auto buf_q_values = q_values.request();
    auto buf_scales = scales.request();
    auto buf_dq_consts = dq_consts.request();

    float* dq_values_ptr = (float*) buf_dq_values.ptr;
    int8_t* q_values_ptr = (int8_t*) buf_q_values.ptr;
    float* scales_ptr = (float*) buf_scales.ptr;
    float* dq_consts_ptr = (float*) buf_dq_consts.ptr;

    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_values_ptr, q_values_ptr, size, scales_ptr, dq_consts_ptr}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(lowerBound, upperBound);
    strategy.quantize_grouped(input, group_size, group_shift_amount);
}

void quantization::standard_8_bit_quantization_grouped_parallel_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    py::array_t<float> scales,
    py::array_t<float> dq_consts,
    int group_size,
    int group_shift_amount,
    float lowerBound,
    float upperBound
) {
    int size = (int) dq_values.size();

    auto buf_dq_values = dq_values.request();
    auto buf_q_values = q_values.request();
    auto buf_scales = scales.request();
    auto buf_dq_consts = dq_consts.request();

    float* dq_values_ptr = (float*) buf_dq_values.ptr;
    int8_t* q_values_ptr = (int8_t*) buf_q_values.ptr;
    float* scales_ptr = (float*) buf_scales.ptr;
    float* dq_consts_ptr = (float*) buf_dq_consts.ptr;

    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_values_ptr, q_values_ptr, size, scales_ptr, dq_consts_ptr}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(lowerBound, upperBound);
    strategy.quantize_grouped_parallel(input, group_size, group_shift_amount);
}