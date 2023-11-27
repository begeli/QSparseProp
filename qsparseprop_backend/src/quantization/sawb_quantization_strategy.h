#ifndef QSPARSEPROP_SAWB_QUANTIZATION_STRATEGY_H
#define QSPARSEPROP_SAWB_QUANTIZATION_STRATEGY_H

#include "quantization_strategy.h"
#include "stochastic_quantization.h"

namespace quantization {
    class SAWBQuantizationStrategy : public StochasticQuantization {
    public:
        SAWBQuantizationStrategy(float lowerBound, float upperBound, float weight1, float weight2)
            : lowerBound(lowerBound), upperBound(upperBound), c1(weight1), c2(weight2) {}

    protected:
        float lowerBound;
        float upperBound;
        float c1;
        float c2;
    };
}

#endif //QSPARSEPROP_SAWB_QUANTIZATION_STRATEGY_H
