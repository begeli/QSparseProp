#ifndef QSPARSEPROP_STANDARD_QUANTIZATION_STRATEGY_H
#define QSPARSEPROP_STANDARD_QUANTIZATION_STRATEGY_H

#include "quantization_strategy.h"
#include "stochastic_quantization.h"

namespace quantization {
    class StandardQuantizationStrategy: public StochasticQuantization {
    public:
        StandardQuantizationStrategy(float lowerBound, float upperBound)
            : lowerBound(lowerBound), upperBound(upperBound) {}

    protected:
        float lowerBound;
        float upperBound;
    };
}

#endif //QSPARSEPROP_STANDARD_QUANTIZATION_STRATEGY_H
