#ifndef QSPARSEPROP_LUQ16_STRATEGY_H
#define QSPARSEPROP_LUQ16_STRATEGY_H

#include "luq_strategy.h"
#include "quantization8_strategy.h"

namespace quantization {
    class LUQ8Strategy: public Quantization8Strategy, public LUQStrategy {
    public:
        explicit LUQ8Strategy(float bits)
            : LUQStrategy(bits, powf(2.0f, -(powf(2.0f, bits) - 1))) {}

        void quantize_scalar(union Quantization_Input<int8_t>& input) override {}
        void quantize(union Quantization_Input<int8_t>& input) override;
        void quantize_parallel(union Quantization_Input<int8_t>& input) override;
        void quantize_grouped(union Quantization_Input<int8_t>& input, int group_size, int group_shift_amount) override;
        void quantize_grouped_parallel(
            union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount
        ) override;

        void restore(union Quantization_Input<int8_t>& input) override;
        void restore_parallel(union Quantization_Input<int8_t>& input) override;
        void restore_grouped(union Quantization_Input<int8_t>& input, int group_size, int group_shift_amount) override;
        void restore_grouped_parallel(
            union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount
        ) override;

        float get_threshold_denom() override;
    };
}

#endif //QSPARSEPROP_LUQ16_STRATEGY_H
