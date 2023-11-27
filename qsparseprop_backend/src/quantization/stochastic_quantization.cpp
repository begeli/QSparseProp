#include "stochastic_quantization.h"

quantization::StochasticQuantization::StochasticQuantization() {
    // Initialize the random numbers per vector, using the hardware random number generator
    avx_xorshift128plus_init(
        get_random_uint64(),
        get_random_uint64(),
        avx_random_key1,
        avx_random_key2
    );

    avx512_xorshift128plus_init(
        get_random_uint64(),
        get_random_uint64(),
        avx512_random_key1,
        avx512_random_key2
    );

    // Allocate sufficient keys for each thread
    const int num_of_threads = get_OpenMP_threads();
    avx_random_key1_perthread = new __m256i[num_of_threads];
    avx_random_key2_perthread = new __m256i[num_of_threads];
    avx512_random_key1_perthread = new __m512i[num_of_threads];
    avx512_random_key2_perthread = new __m512i[num_of_threads];
    // Then initialize random keys that could be used per thread.
    for (int i = 0; i < num_of_threads; i++) {
        avx_xorshift128plus_init(
            get_random_uint64(),
            get_random_uint64(),
            avx_random_key1_perthread[i],
            avx_random_key2_perthread[i]
        );

        avx512_xorshift128plus_init(
            get_random_uint64(),
            get_random_uint64(),
            avx512_random_key1_perthread[i],
            avx512_random_key2_perthread[i]
        );
    }
}

quantization::StochasticQuantization::~StochasticQuantization() {
    delete [] avx_random_key1_perthread;
    delete [] avx_random_key2_perthread;
    delete [] avx512_random_key1_perthread;
    delete [] avx512_random_key2_perthread;
}

uint64_t quantization::StochasticQuantization::get_random_uint64() {
    unsigned long long rnd1;
    int ret = 0;
    while (ret == 0) {
        ret = _rdrand64_step(&rnd1);
    }
    return (uint64_t) rnd1;
}
