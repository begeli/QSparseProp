#ifndef QSPARSEPROP_SAWB_QUANTIZATION8_STRATEGY_H
#define QSPARSEPROP_SAWB_QUANTIZATION8_STRATEGY_H

#include "sawb_quantization_strategy.h"
#include "quantization8_strategy.h"

namespace quantization {
    class SAWBQuantization8Strategy: public Quantization8Strategy, public SAWBQuantizationStrategy {
    public:
        SAWBQuantization8Strategy(float lowerBound, float upperBound, float weight1, float weight2)
            : SAWBQuantizationStrategy(lowerBound, upperBound, weight1, weight2) {}

        void quantize_scalar(union Quantization_Input<int8_t>& input) override;
        void quantize(union Quantization_Input<int8_t>& input) override;
        void quantize_parallel(union Quantization_Input<int8_t>& input) override;
        void quantize_grouped(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) override;
        void quantize_grouped_parallel(
            union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount
        ) override;

        void restore(union Quantization_Input<int8_t>& input) override;
        void restore_parallel(union Quantization_Input<int8_t>& input) override;
        void restore_grouped(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) override;
        void restore_grouped_parallel(
            union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount
        ) override;
    };
}

#endif //QSPARSEPROP_SAWB_QUANTIZATION8_STRATEGY_H
