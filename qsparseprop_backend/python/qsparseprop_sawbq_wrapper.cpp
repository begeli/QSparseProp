#include "qsparseprop_sawbq_wrapper.h"

float quantization::sawb_8_bit_scalar_quantization_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    float lowerBound,
    float upperBound,
    float weight1,
    float weight2
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

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(lowerBound, upperBound, weight1, weight2);
    strategy.quantize_scalar(input);

    return input.std_quantization_input.scale;
}

float quantization::sawb_8_bit_quantization_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    float lowerBound,
    float upperBound,
    float weight1,
    float weight2
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

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(lowerBound, upperBound, weight1, weight2);
    strategy.quantize(input);

    return input.std_quantization_input.scale;
}

float quantization::sawb_8_bit_quantization_parallel_wrapper(
    py::array_t<float> dq_values,
    py::array_t<int8_t> q_values,
    float lowerBound,
    float upperBound,
    float weight1,
    float weight2
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

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(lowerBound, upperBound, weight1, weight2);
    strategy.quantize_parallel(input);

    return input.std_quantization_input.scale;
}

void quantization::sawb_8_bit_quantization_grouped_wrapper(
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

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(lowerBound, upperBound, weight1, weight2);
    strategy.quantize_grouped(input, group_size, group_shift_amount);
}

void quantization::sawb_8_bit_quantization_grouped_parallel_wrapper(
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

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(lowerBound, upperBound, weight1, weight2);
    strategy.quantize_grouped_parallel(input, group_size, group_shift_amount);
}