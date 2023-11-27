#ifndef QSPARSEPROP_QUANTIZATION_STRATEGY_H
#define QSPARSEPROP_QUANTIZATION_STRATEGY_H

#include <cstdint>
#include <cmath>
#include "src/conf/constants.h"
#include "src/utils/simdmath.h"

namespace quantization {
    template<typename T>
    struct Std_Quantization_Input {
        float* dq_values;
        T* q_values;
        int size;
        float scale;
        float dequantization_const;
    };

    template<typename T>
    struct Std_Grouped_Quantization_Input {
        float* dq_values;
        T* q_values;
        int size;
        float* scale;
        float* dequantization_const;
    };

    template<typename T>
    struct LUQ_Quantization_Input {
        float* dq_values;
        T* q_values;
        int size;
        float scale;
        float dequantization_const;
        int mask_count;
        __mmask16* signs;
    };

    template<typename T>
    struct LUQ_Grouped_Quantization_Input {
        float* dq_values;
        T* q_values;
        int size;
        float* scale;
        float* dequantization_const;
        int mask_count;
        __mmask16* signs;
    };

    template<typename T>
    union Quantization_Input {
        Std_Quantization_Input<T> std_quantization_input;
        Std_Grouped_Quantization_Input<T> std_grouped_input;
        LUQ_Quantization_Input<T> luq_quantization_input;
        LUQ_Grouped_Quantization_Input<T> luq_grouped_input;
    };

    template<typename T>
    class QuantizationStrategy {
    public:
        QuantizationStrategy() = default;

        virtual void quantize_scalar(union Quantization_Input<T>& input) = 0;
        virtual void quantize(union Quantization_Input<T>& input) = 0;
        virtual void quantize_parallel(union Quantization_Input<T>& input) = 0;
        virtual void quantize_grouped(union Quantization_Input<T>& input, int qgroup_size, int qgroup_shift_amount) = 0;
        virtual void quantize_grouped_parallel(
            union Quantization_Input<T>& input, int qgroup_size, int qgroup_shift_amount
        ) = 0;

        virtual void restore(union Quantization_Input<T>& input) = 0;
        virtual void restore_parallel(union Quantization_Input<T>& input) = 0;
        virtual void restore_grouped(union Quantization_Input<T>& input, int qgroup_size, int qgroup_shift_amount) = 0;
        virtual void restore_grouped_parallel(
            union Quantization_Input<T>& input, int qgroup_size, int qgroup_shift_amount
        ) = 0;
    };
}

#endif //QSPARSEPROP_QUANTIZATION_STRATEGY_H
