#ifndef QSPARSEPROP_LUQ_STRATEGY_H
#define QSPARSEPROP_LUQ_STRATEGY_H

#include "quantization_strategy.h"
#include "stochastic_quantization.h"

namespace quantization {
    class LUQStrategy : public StochasticQuantization {
    public:
        LUQStrategy(float bits, float thresholdDenom): bits(bits), threshold_denom(thresholdDenom) {}

        virtual float get_threshold_denom() = 0;
    protected:
        float bits;
        float threshold_denom;
    };
}

#endif //QSPARSEPROP_LUQ_STRATEGY_H
