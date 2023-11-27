#ifndef QSPARSEPROP_DITHERED_QUANTIZATION_STRATEGY_H
#define QSPARSEPROP_DITHERED_QUANTIZATION_STRATEGY_H

#include "quantization_strategy.h"
#include "stochastic_quantization.h"

namespace quantization {
    class DitheredQuantizationStrategy : public StochasticQuantization {
    public:
        explicit DitheredQuantizationStrategy(float dithered_scale): dithered_scale(dithered_scale) {}

    protected:
        float dithered_scale;
    };
}

#endif //QSPARSEPROP_DITHERED_QUANTIZATION_STRATEGY_H
