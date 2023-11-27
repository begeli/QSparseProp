#ifndef QSPARSEPROP_STOCHASTIC_QUANTIZATION_H
#define QSPARSEPROP_STOCHASTIC_QUANTIZATION_H

#include <immintrin.h>
#include "src/utils/simdxorshift128plus.h"
#include "src/utils/openmp_utils.h"

namespace quantization {
    class StochasticQuantization {
    public:
        StochasticQuantization();
        ~StochasticQuantization();

    protected:
        __m256i avx_random_key1;
        __m256i avx_random_key2;
        __m256i * avx_random_key1_perthread;
        __m256i * avx_random_key2_perthread;
        __m512i avx512_random_key1;
        __m512i avx512_random_key2;
        __m512i * avx512_random_key1_perthread;
        __m512i * avx512_random_key2_perthread;

        // Return a random number between 0 and 1
        inline float get_random_float() {
            unsigned int i_rnd;
            int ret = 0;
            while (ret == 0) {
                ret = _rdrand32_step(&i_rnd);
            }
            const float f_rnd = (float) i_rnd;
            return f_rnd * (1.0f / 4294967296.0f);
        }
    private:
        static uint64_t get_random_uint64();
    };
}

#endif //QSPARSEPROP_STOCHASTIC_QUANTIZATION_H
