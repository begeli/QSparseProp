#ifndef QSPARSEPROP_DITHERED_QUANTIZATION8_STRATEGY_H
#define QSPARSEPROP_DITHERED_QUANTIZATION8_STRATEGY_H

#include "dithered_quantization_strategy.h"
#include "quantization8_strategy.h"

namespace quantization {
    class DitheredQuantization8Strategy: public Quantization8Strategy, public DitheredQuantizationStrategy {
    public:
        explicit DitheredQuantization8Strategy(float dithered_scale)
            : DitheredQuantizationStrategy(dithered_scale) {}

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

#endif //QSPARSEPROP_DITHERED_QUANTIZATION8_STRATEGY_H
