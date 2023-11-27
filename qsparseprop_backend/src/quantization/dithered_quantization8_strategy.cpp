#include "dithered_quantization8_strategy.h"

#include <iostream>

void quantization::DitheredQuantization8Strategy::quantize_scalar(union Quantization_Input<int8_t>& input) {
    /*const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& step_size = input.std_quantization_input.scale;
    const float* signal = input.std_quantization_input.noise;

    __m512 acc_sq_val_vec_0 = _mm512_setzero_ps();
    __m512 acc_sq_val_vec_1 = _mm512_setzero_ps();
    __m512 acc_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_mean_vec_1 = _mm512_setzero_ps();

    // Compute standard deviation - Expand the standard deviation formula to compute all constituents at the same time
    int idx;
    for (idx = 0; idx < size - 31; idx += 32) {
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        const __m512 sq_val_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_val_vec_0);
        const __m512 sq_val_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_val_vec_1);
        const __m512 mean_vec_0 = _mm512_add_ps(val_vec_0, acc_mean_vec_0);
        const __m512 mean_vec_1 = _mm512_add_ps(val_vec_1, acc_mean_vec_1);

        acc_sq_val_vec_0 = sq_val_vec_0;
        acc_sq_val_vec_1 = sq_val_vec_1;
        acc_mean_vec_0 = mean_vec_0;
        acc_mean_vec_1 = mean_vec_1;
    }
    const float acc_sq_val_0 = _mm512_haddf32_ss(acc_sq_val_vec_0);
    const float acc_sq_val_1 = _mm512_haddf32_ss(acc_sq_val_vec_1);
    const float acc_mean_0 = _mm512_haddf32_ss(acc_mean_vec_0);
    const float acc_mean_1 = _mm512_haddf32_ss(acc_mean_vec_1);
    float acc_sq_val = acc_sq_val_0 + acc_sq_val_1;
    float acc_mean = acc_mean_0 + acc_mean_1;

    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float sq = val * val;
        const float sq_sum = acc_sq_val + sq;
        const float mean_sum = acc_mean + val;

        // Store
        acc_sq_val = sq_sum;
        acc_mean = mean_sum;
    }

    const float rcp_size = 1.0f / (float) size;
    const float mean = acc_mean * rcp_size;
    //const float mean = acc_mean / size;
    const float mean_sq = mean * mean;
    const float sum_sq_div_size = acc_sq_val * rcp_size;
    //const float sum_sq_div_size = acc_sq_val / size;
    const float std = sqrt(sum_sq_div_size - mean_sq);
    float rcp_step_size;

    step_size = dithered_scale * std;
    rcp_step_size = 1.0f / step_size;

    const __m512 step_size_vec = _mm512_set1_ps(step_size);
    const __m512 rcp_step_size_vec = _mm512_set1_ps(rcp_step_size);
    idx = 0;
    for (; idx < size - 31; idx += 32) {
        // Load
        const float* value_offset = values + idx;
        const __m512 val_vec_0 = _mm512_loadu_ps(value_offset);
        const __m512 val_vec_1 = _mm512_loadu_ps(value_offset + 16);

        // Generate randomness
        const float* signal_offset = signal + idx;
        const __m512 signal_0 = _mm512_loadu_ps(signal_offset);
        const __m512 signal_1 = _mm512_loadu_ps(signal_offset + 16);

        const __m512 rnd_val_vec_0 = _mm512_fmadd_ps(step_size_vec, signal_0, val_vec_0);
        const __m512 rnd_val_vec_1 = _mm512_fmadd_ps(step_size_vec, signal_1, val_vec_1);

        const __m512 normalized_vec_0 = _mm512_fmadd_ps(rnd_val_vec_0, rcp_step_size_vec, _mm512_0_5_ps);
        const __m512 normalized_vec_1 = _mm512_fmadd_ps(rnd_val_vec_1, rcp_step_size_vec, _mm512_0_5_ps);

        const __m512 rounded_vec_0 = _mm512_floor_ps(normalized_vec_0);
        const __m512 rounded_vec_1 = _mm512_floor_ps(normalized_vec_1);

        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(rounded_vec_0); // try floor or some other function, if this doesn't work
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(rounded_vec_1);

        // Pack
        // Step 1: break the 512bit vectors into two
        const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
        const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
        const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
        const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

        // Step 3: Pack the vectors into a single vector
        __m256i pack8;
        Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

        // Step 4: Store the combined vectors in the result array
        _mm256_storeu_si256((__m256i *) (result + idx), pack8);
    }

    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Generate randomness
        const float rnd = signal[idx] * step_size;

        // Compute
        const float rnd_val = val + rnd;
        const float normalized_val_0 = rnd_val * rcp_step_size;
        const float normalized_val_1 = normalized_val_0 + 0.5f;
        const int8_t quantized_val = (int8_t) floorf(normalized_val_1);

        result[idx] = quantized_val;
    }*/
}

void quantization::DitheredQuantization8Strategy::quantize(
    union Quantization_Input<int8_t>& input
) {
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& step_size = input.std_quantization_input.scale;
    //const float* signal = input.std_quantization_input.noise;

    __m512 acc_sq_val_vec_0 = _mm512_setzero_ps();
    __m512 acc_sq_val_vec_1 = _mm512_setzero_ps();
    __m512 acc_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_mean_vec_1 = _mm512_setzero_ps();

    // Compute standard deviation - Expand the standard deviation formula to compute all constituents at the same time
    int idx;
    for (idx = 0; idx < size - 31; idx += 32) {
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        const __m512 sq_val_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_val_vec_0);
        const __m512 sq_val_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_val_vec_1);
        const __m512 mean_vec_0 = _mm512_add_ps(val_vec_0, acc_mean_vec_0);
        const __m512 mean_vec_1 = _mm512_add_ps(val_vec_1, acc_mean_vec_1);

        acc_sq_val_vec_0 = sq_val_vec_0;
        acc_sq_val_vec_1 = sq_val_vec_1;
        acc_mean_vec_0 = mean_vec_0;
        acc_mean_vec_1 = mean_vec_1;
    }
    const float acc_sq_val_0 = _mm512_haddf32_ss(acc_sq_val_vec_0);
    const float acc_sq_val_1 = _mm512_haddf32_ss(acc_sq_val_vec_1);
    const float acc_mean_0 = _mm512_haddf32_ss(acc_mean_vec_0);
    const float acc_mean_1 = _mm512_haddf32_ss(acc_mean_vec_1);
    float acc_sq_val = acc_sq_val_0 + acc_sq_val_1;
    float acc_mean = acc_mean_0 + acc_mean_1;

    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float sq = val * val;
        const float sq_sum = acc_sq_val + sq;
        const float mean_sum = acc_mean + val;

        // Store
        acc_sq_val = sq_sum;
        acc_mean = mean_sum;
    }

    //const float rcp_size = 1.0f / (float) size;
    //const float mean = acc_mean * rcp_size;
    const float mean = acc_mean / size;
    const float mean_sq = mean * mean;
    //const float sum_sq_div_size = acc_sq_val * rcp_size;
    const float sum_sq_div_size = acc_sq_val / size;
    const float std = sqrt(sum_sq_div_size - mean_sq);
    //float rcp_step_size;

    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //if (std >= EPS) {
        step_size = dithered_scale * std;
        //rcp_step_size = 1.0f / step_size;
    //} else {
        // makes random number 0, before the function call is over, make sure to set this to the only value in the array.
    //    step_size = 0.0f;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //    if (std::abs(values[0]) <= EPS) {
    //        rcp_step_size = 1.0f;
    //    } else {
    //        rcp_step_size = 1.0f / values[0];
    //    }
    //}

    const __m512 step_size_vec = _mm512_set1_ps(step_size);
    //const __m512 rcp_step_size_vec = _mm512_set1_ps(rcp_step_size);
    idx = 0;
    for (; idx < size - 31; idx += 32) {
        // Load
        const float* value_offset = values + idx;
        const __m512 val_vec_0 = _mm512_loadu_ps(value_offset);
        const __m512 val_vec_1 = _mm512_loadu_ps(value_offset + 16);

        // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const __m512 rnd_0 = _mm512_setzero_ps();
        const __m512 rnd_1 = _mm512_setzero_ps();

        const __m512 rnd_val_vec_0 = _mm512_add_ps(val_vec_0, rnd_0);
        const __m512 rnd_val_vec_1 = _mm512_add_ps(val_vec_1, rnd_1);
#else
        /*const float* signal_offset = signal + idx;
        const __m512 signal_0 = _mm512_loadu_ps(signal_offset);
        const __m512 signal_1 = _mm512_loadu_ps(signal_offset + 16);

        const __m512 rnd_val_vec_0 = _mm512_fmadd_ps(step_size_vec, signal_0, val_vec_0);
        const __m512 rnd_val_vec_1 = _mm512_fmadd_ps(step_size_vec, signal_1, val_vec_1);*/

        const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

        const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
        const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

        const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
        const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

        const __m512 rnd_scaled_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_scaled_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);

        const __m512 rnd_val_vec_0 = _mm512_fmadd_ps(step_size_vec, rnd_scaled_0, val_vec_0);
        const __m512 rnd_val_vec_1 = _mm512_fmadd_ps(step_size_vec, rnd_scaled_1, val_vec_1);
#endif
        //const __m512 normalized_vec_0 = _mm512_fmadd_ps(rnd_val_vec_0, rcp_step_size_vec, _mm512_0_5_ps);
        //const __m512 normalized_vec_1 = _mm512_fmadd_ps(rnd_val_vec_1, rcp_step_size_vec, _mm512_0_5_ps);
        const __m512 normalized_tmp_0 = _mm512_div_ps(rnd_val_vec_0, step_size_vec);
        const __m512 normalized_tmp_1 = _mm512_div_ps(rnd_val_vec_1, step_size_vec);
        const __m512 normalized_vec_0 = _mm512_add_ps(normalized_tmp_0, _mm512_0_5_ps);
        const __m512 normalized_vec_1 = _mm512_add_ps(normalized_tmp_1, _mm512_0_5_ps);

        const __m512 rounded_vec_0 = _mm512_floor_ps(normalized_vec_0);
        const __m512 rounded_vec_1 = _mm512_floor_ps(normalized_vec_1);

        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(rounded_vec_0); // try floor or some other function, if this doesn't work
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(rounded_vec_1);

        // Pack
        // Step 1: break the 512bit vectors into two
        const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
        const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
        const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
        const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

        // Step 3: Pack the vectors into a single vector
        __m256i pack8;
        Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

        // Step 4: Store the combined vectors in the result array
        _mm256_storeu_si256((__m256i *) (result + idx), pack8);
    }

    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const float rnd = 0;
#else
        //const float rnd = signal[idx] * step_size;

        const float rnd_0_to_1 = get_random_float();
        const float rnd_scaled = rnd_0_to_1 - 0.5f;
        const float rnd = rnd_scaled * step_size;
#endif
        // Compute
        const float rnd_val = val + rnd;
        //const float normalized_val_0 = rnd_val * rcp_step_size;
        const float normalized_val_0 = rnd_val / step_size;
        const float normalized_val_1 = normalized_val_0 + 0.5f;
        const int8_t quantized_val = (int8_t) floorf(normalized_val_1);

        result[idx] = quantized_val;
    }

    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //if (std <= EPS) {
    //    step_size = values[0];
    //}
}

void quantization::DitheredQuantization8Strategy::quantize_parallel(
    union Quantization_Input<int8_t>& input
) {
#if defined(_OPENMP)
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& step_size = input.std_quantization_input.scale;
    //const float* signal = input.std_quantization_input.noise;

    const int thread_count = get_OpenMP_threads();
    __m512* acc_sq_val_vecs_0 = new __m512[thread_count];
    __m512* acc_sq_val_vecs_1 = new __m512[thread_count];
    __m512* acc_mean_vecs_0 = new __m512[thread_count];
    __m512* acc_mean_vecs_1 = new __m512[thread_count];
    for (int thread = 0; thread < thread_count; thread++) {
        acc_sq_val_vecs_0[thread] = _mm512_setzero_ps();
        acc_sq_val_vecs_1[thread] = _mm512_setzero_ps();
        acc_mean_vecs_0[thread] = _mm512_setzero_ps();
        acc_mean_vecs_1[thread] = _mm512_setzero_ps();
    }

    // Compute standard deviation
    int idx;
    #pragma omp parallel for default(none) shared(size, values, acc_sq_val_vecs_0, acc_sq_val_vecs_1, acc_mean_vecs_0, acc_mean_vecs_1)
    for (idx = 0; idx < size - 31; idx += 32) {
        const int tid = get_OpenMP_thread();

        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        const __m512 sq_val_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_val_vecs_0[tid]);
        const __m512 sq_val_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_val_vecs_1[tid]);
        const __m512 mean_vec_0 = _mm512_add_ps(val_vec_0, acc_mean_vecs_0[tid]);
        const __m512 mean_vec_1 = _mm512_add_ps(val_vec_1, acc_mean_vecs_1[tid]);

        acc_sq_val_vecs_0[tid] = sq_val_vec_0;
        acc_sq_val_vecs_1[tid] = sq_val_vec_1;
        acc_mean_vecs_0[tid] = mean_vec_0;
        acc_mean_vecs_1[tid] = mean_vec_1;
    }

    __m512 acc_sq_val_vec_0 = _mm512_setzero_ps();
    __m512 acc_sq_val_vec_1 = _mm512_setzero_ps();
    __m512 acc_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_mean_vec_1 = _mm512_setzero_ps();
    for (int thread = 0; thread < thread_count; thread++) {
        acc_sq_val_vec_0 = _mm512_add_ps(acc_sq_val_vec_0, acc_sq_val_vecs_0[thread]);
        acc_sq_val_vec_1 = _mm512_add_ps(acc_sq_val_vec_1, acc_sq_val_vecs_1[thread]);
        acc_mean_vec_0 = _mm512_add_ps(acc_mean_vec_0, acc_mean_vecs_0[thread]);
        acc_mean_vec_1 = _mm512_add_ps(acc_mean_vec_1, acc_mean_vecs_1[thread]);
    }
    delete [] acc_sq_val_vecs_0;
    delete [] acc_sq_val_vecs_1;
    delete [] acc_mean_vecs_0;
    delete [] acc_mean_vecs_1;

    const float acc_sq_val_0 = _mm512_haddf32_ss(acc_sq_val_vec_0);
    const float acc_sq_val_1 = _mm512_haddf32_ss(acc_sq_val_vec_1);
    const float acc_mean_0 = _mm512_haddf32_ss(acc_mean_vec_0);
    const float acc_mean_1 = _mm512_haddf32_ss(acc_mean_vec_1);
    float acc_sq_val = acc_sq_val_0 + acc_sq_val_1;
    float acc_mean = acc_mean_0 + acc_mean_1;

    idx = (size / 32) * 32;
    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float sq = val * val;
        const float sq_sum = acc_sq_val + sq;
        const float mean_sum = acc_mean + val;

        // Store
        acc_sq_val = sq_sum;
        acc_mean = mean_sum;
    }

    //const float rcp_size = 1.0f / (float) size;
    //const float mean = acc_mean * rcp_size;
    const float mean = acc_mean / size;
    const float mean_sq = mean * mean;
    //const float sum_sq_div_size = acc_sq_val * rcp_size;
    const float sum_sq_div_size = acc_sq_val / size;
    const float std = sqrt(sum_sq_div_size - mean_sq);
    //float rcp_step_size;
    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //if (std >= EPS) {
        step_size = dithered_scale * std;
        //rcp_step_size = 1.0f / step_size;
    //} else {
        // makes random number 0, before the function call is over, make sure to set this to the only value in the array.
    //    step_size = 0.0f;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //    if (std::abs(values[0]) <= EPS) {
    //        rcp_step_size = 1.0f;
    //    } else {
    //        rcp_step_size = 1.0f / values[0];
    //    }
    //}

    const __m512 step_size_vec = _mm512_set1_ps(step_size);
    //const __m512 rcp_step_size_vec = _mm512_set1_ps(rcp_step_size);
    #pragma omp parallel for default(none) shared(size, values, /*rcp_step_size_vec,*/ _mm512_0_5_ps, result, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, step_size_vec, avx512_random_key1_perthread, avx512_random_key2_perthread)
    for (idx = 0; idx < size - 31; idx += 32) {
        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const __m512 rnd_0 = _mm512_setzero_ps();
        const __m512 rnd_1 = _mm512_setzero_ps();

        const __m512 rnd_val_vec_0 = _mm512_add_ps(val_vec_0, rnd_0);
        const __m512 rnd_val_vec_1 = _mm512_add_ps(val_vec_1, rnd_1);
#else
        /*const float* signal_offset = signal + idx;
        const __m512 signal_0 = _mm512_loadu_ps(signal_offset);
        const __m512 signal_1 = _mm512_loadu_ps(signal_offset + 16);

        const __m512 rnd_val_vec_0 = _mm512_fmadd_ps(step_size_vec, signal_0, val_vec_0);
        const __m512 rnd_val_vec_1 = _mm512_fmadd_ps(step_size_vec, signal_1, val_vec_1);*/

        const int tid = get_OpenMP_thread();

        const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1_perthread[tid], avx512_random_key2_perthread[tid]);

        const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
        const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

        const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
        const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

        const __m512 rnd_scaled_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_scaled_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);

        const __m512 rnd_val_vec_0 = _mm512_fmadd_ps(step_size_vec, rnd_scaled_0, val_vec_0);
        const __m512 rnd_val_vec_1 = _mm512_fmadd_ps(step_size_vec, rnd_scaled_1, val_vec_1);
#endif
        //const __m512 normalized_vec_0 = _mm512_fmadd_ps(rnd_val_vec_0, rcp_step_size_vec, _mm512_0_5_ps);
        //const __m512 normalized_vec_1 = _mm512_fmadd_ps(rnd_val_vec_1, rcp_step_size_vec, _mm512_0_5_ps);
        const __m512 normalized_tmp_0 = _mm512_div_ps(rnd_val_vec_0, step_size_vec);
        const __m512 normalized_tmp_1 = _mm512_div_ps(rnd_val_vec_1, step_size_vec);
        const __m512 normalized_vec_0 = _mm512_add_ps(normalized_tmp_0, _mm512_0_5_ps);
        const __m512 normalized_vec_1 = _mm512_add_ps(normalized_tmp_1, _mm512_0_5_ps);

        const __m512 rounded_vec_0 = _mm512_floor_ps(normalized_vec_0);
        const __m512 rounded_vec_1 = _mm512_floor_ps(normalized_vec_1);

        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(rounded_vec_0); // try floor or some other function, if this doesn't work
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(rounded_vec_1);

        // Pack
        // Step 1: break the 512bit vectors into two
        const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
        const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
        const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
        const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

        // Step 3: Pack the vectors into a single vector
        __m256i pack8;
        Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

        // Step 4: Store the combined vectors in the result array
        _mm256_storeu_si256((__m256i *) (result + idx), pack8);
    }

    idx = (size / 32) * 32;
    #pragma omp parallel for default(none) shared(idx, size, values, step_size, /*rcp_step_size,*/ result)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // Load
        const float val = values[idx2];

        // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const float rnd = 0;
#else
        //const float rnd = signal[idx] * step_size;

        const float rnd_0_to_1 = get_random_float();
        const float rnd_scaled = rnd_0_to_1 - 0.5f;
        const float rnd = rnd_scaled * step_size;
#endif
        // Compute
        const float rnd_val = val + rnd;
        //const float normalized_val_0 = rnd_val * rcp_step_size;
        const float normalized_val_0 = rnd_val / step_size;
        const float normalized_val_1 = normalized_val_0 + 0.5f;
        const int8_t quantized_val = (int8_t) floorf(normalized_val_1);

        result[idx2] = quantized_val;
    }

    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    //if (std <= EPS) {
    //    step_size = values[0];
    //}
#else
    quantization::DitheredQuantization8Strategy::quantize(input);
#endif
}

void quantization::DitheredQuantization8Strategy::quantize_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* step_size = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
            ? (size % qgroup_size) == 0
                ? qgroup_size
                : size % qgroup_size
            : qgroup_size;
        const int group_start = group * qgroup_size;
        const float* group_values = values + group_start;
        int8_t* group_result = result + group_start;

        __m512 acc_sq_val_vec_0 = _mm512_setzero_ps();
        __m512 acc_sq_val_vec_1 = _mm512_setzero_ps();
        __m512 acc_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_mean_vec_1 = _mm512_setzero_ps();
        // Compute standard deviation - Expand the standard deviation formula to compute all constituents at the same time
        int idx;
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const __m512 val_vec_0 = _mm512_loadu_ps(group_values + idx);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_values + idx + 16);

            const __m512 sq_val_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_val_vec_0);
            const __m512 sq_val_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_val_vec_1);
            const __m512 mean_vec_0 = _mm512_add_ps(val_vec_0, acc_mean_vec_0);
            const __m512 mean_vec_1 = _mm512_add_ps(val_vec_1, acc_mean_vec_1);

            acc_sq_val_vec_0 = sq_val_vec_0;
            acc_sq_val_vec_1 = sq_val_vec_1;
            acc_mean_vec_0 = mean_vec_0;
            acc_mean_vec_1 = mean_vec_1;
        }
        const float acc_sq_val_0 = _mm512_haddf32_ss(acc_sq_val_vec_0);
        const float acc_sq_val_1 = _mm512_haddf32_ss(acc_sq_val_vec_1);
        const float acc_mean_0 = _mm512_haddf32_ss(acc_mean_vec_0);
        const float acc_mean_1 = _mm512_haddf32_ss(acc_mean_vec_1);
        float acc_sq_val = acc_sq_val_0 + acc_sq_val_1;
        float acc_mean = acc_mean_0 + acc_mean_1;

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float sq = val * val;
            const float sq_sum = acc_sq_val + sq;
            const float mean_sum = acc_mean + val;

            // Store
            acc_sq_val = sq_sum;
            acc_mean = mean_sum;
        }

        const float rcp_size = 1.0f / (float) group_size;
        const float mean = acc_mean * rcp_size;
        const float mean_sq = mean * mean;
        const float sum_sq_div_size = acc_sq_val * rcp_size;
        const float std = sqrt(sum_sq_div_size - mean_sq);
        float rcp_step_size;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std >= EPS) {
            step_size[group] = dithered_scale * std;
            rcp_step_size = 1.0f / step_size[group];
        } else {
            // makes random number 0, before the function call is over, make sure to set this to the only value in the array.
            step_size[group] = 0.0f;
            // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
            if (std::abs(group_values[0]) <= EPS) {
                rcp_step_size = 1.0f;
            } else {
                rcp_step_size = 1.0f / group_values[0];
            }
        }
        dequantization_const[group] = 0.0f;

        const __m512 step_size_vec = _mm512_set1_ps(step_size[group]);
        const __m512 rcp_step_size_vec = _mm512_set1_ps(rcp_step_size);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const __m512 val_vec_0 = _mm512_loadu_ps(group_values + idx);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_values + idx + 16);

            // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const __m512 rnd_0 = _mm512_setzero_ps();
            const __m512 rnd_1 = _mm512_setzero_ps();
#else
            const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

            const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
            const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

            const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
            const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

            const __m512 rnd_scaled_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_scaled_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);

            const __m512 rnd_0 = _mm512_mul_ps(step_size_vec, rnd_scaled_0);
            const __m512 rnd_1 = _mm512_mul_ps(step_size_vec, rnd_scaled_1);
#endif
            const __m512 rnd_val_vec_0 = _mm512_add_ps(val_vec_0, rnd_0);
            const __m512 rnd_val_vec_1 = _mm512_add_ps(val_vec_1, rnd_1);

            const __m512 normalized_vec_0 = _mm512_fmadd_ps(rnd_val_vec_0, rcp_step_size_vec, _mm512_0_5_ps);
            const __m512 normalized_vec_1 = _mm512_fmadd_ps(rnd_val_vec_1, rcp_step_size_vec, _mm512_0_5_ps);

            const __m512i quantized_vec_0 = _mm512_cvttps_epi32(normalized_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvttps_epi32(normalized_vec_1);

            // Pack
            // Step 1: break the 512bit vectors into two
            const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
            const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
            const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
            const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

            // Step 3: Pack the vectors into a single vector
            __m256i pack8;
            Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

            // Step 4: Store the combined vectors in the result array
            _mm256_storeu_si256((__m256i *) (group_result + idx), pack8);
        }

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float rnd = 0;
#else
            const float rnd_0_to_1 = get_random_float();
            const float rnd_scaled = rnd_0_to_1 - 0.5f;
            const float rnd = rnd_scaled * step_size[group];
#endif
            // Compute
            const float rnd_val = val + rnd;
            const float normalized_val_0 = rnd_val * rcp_step_size;
            const float normalized_val_1 = normalized_val_0 + 0.5f;
            const int8_t quantized_val = (int8_t) floorf(normalized_val_1);

            group_result[idx] = quantized_val;
        }

        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std <= EPS) {
            step_size[group] = group_values[0];
        }
    }
}

void quantization::DitheredQuantization8Strategy::quantize_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* step_size = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, EPS, step_size, dequantization_const, _mm512_0_5_ps, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, avx512_random_key1_perthread, avx512_random_key2_perthread)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const float* group_values = values + group_start;
        int8_t* group_result = result + group_start;

        __m512 acc_sq_val_vec_0 = _mm512_setzero_ps();
        __m512 acc_sq_val_vec_1 = _mm512_setzero_ps();
        __m512 acc_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_mean_vec_1 = _mm512_setzero_ps();
        // Compute standard deviation - Expand the standard deviation formula to compute all constituents at the same time
        int idx;
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const __m512 val_vec_0 = _mm512_loadu_ps(group_values + idx);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_values + idx + 16);

            const __m512 sq_val_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_val_vec_0);
            const __m512 sq_val_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_val_vec_1);
            const __m512 mean_vec_0 = _mm512_add_ps(val_vec_0, acc_mean_vec_0);
            const __m512 mean_vec_1 = _mm512_add_ps(val_vec_1, acc_mean_vec_1);

            acc_sq_val_vec_0 = sq_val_vec_0;
            acc_sq_val_vec_1 = sq_val_vec_1;
            acc_mean_vec_0 = mean_vec_0;
            acc_mean_vec_1 = mean_vec_1;
        }
        const float acc_sq_val_0 = _mm512_haddf32_ss(acc_sq_val_vec_0);
        const float acc_sq_val_1 = _mm512_haddf32_ss(acc_sq_val_vec_1);
        const float acc_mean_0 = _mm512_haddf32_ss(acc_mean_vec_0);
        const float acc_mean_1 = _mm512_haddf32_ss(acc_mean_vec_1);
        float acc_sq_val = acc_sq_val_0 + acc_sq_val_1;
        float acc_mean = acc_mean_0 + acc_mean_1;

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float sq = val * val;
            const float sq_sum = acc_sq_val + sq;
            const float mean_sum = acc_mean + val;

            // Store
            acc_sq_val = sq_sum;
            acc_mean = mean_sum;
        }

        const float rcp_size = 1.0f / (float) group_size;
        const float mean = acc_mean * rcp_size;
        const float mean_sq = mean * mean;
        const float sum_sq_div_size = acc_sq_val * rcp_size;
        const float std = sqrt(sum_sq_div_size - mean_sq);
        float rcp_step_size;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std >= EPS) {
            step_size[group] = dithered_scale * std;
            rcp_step_size = 1.0f / step_size[group];
        } else {
            // makes random number 0, before the function call is over, make sure to set this to the only value in the array.
            step_size[group] = 0.0f;
            // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
            if (std::abs(group_values[0]) <= EPS) {
                rcp_step_size = 1.0f;
            } else {
                rcp_step_size = 1.0f / group_values[0];
            }
        }
        dequantization_const[group] = 0.0f;

        const __m512 step_size_vec = _mm512_set1_ps(step_size[group]);
        const __m512 rcp_step_size_vec = _mm512_set1_ps(rcp_step_size);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const __m512 val_vec_0 = _mm512_loadu_ps(group_values + idx);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_values + idx + 16);

            // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const __m512 rnd_0 = _mm512_setzero_ps();
            const __m512 rnd_1 = _mm512_setzero_ps();
#else
            const int tid = get_OpenMP_thread();

            const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1_perthread[tid], avx512_random_key2_perthread[tid]);

            const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
            const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

            const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
            const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

            const __m512 rnd_scaled_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_scaled_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);

            const __m512 rnd_0 = _mm512_mul_ps(step_size_vec, rnd_scaled_0);
            const __m512 rnd_1 = _mm512_mul_ps(step_size_vec, rnd_scaled_1);
#endif
            const __m512 rnd_val_vec_0 = _mm512_add_ps(val_vec_0, rnd_0);
            const __m512 rnd_val_vec_1 = _mm512_add_ps(val_vec_1, rnd_1);

            const __m512 normalized_vec_0 = _mm512_fmadd_ps(rnd_val_vec_0, rcp_step_size_vec, _mm512_0_5_ps);
            const __m512 normalized_vec_1 = _mm512_fmadd_ps(rnd_val_vec_1, rcp_step_size_vec, _mm512_0_5_ps);

            const __m512i quantized_vec_0 = _mm512_cvttps_epi32(normalized_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvttps_epi32(normalized_vec_1);

            // Pack
            // Step 1: break the 512bit vectors into two
            const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
            const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
            const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
            const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

            // Step 3: Pack the vectors into a single vector
            __m256i pack8;
            Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

            // Step 4: Store the combined vectors in the result array
            _mm256_storeu_si256((__m256i *) (group_result + idx), pack8);
        }

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Generate randomness
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float rnd = 0;
#else
            const float rnd_0_to_1 = get_random_float();
            const float rnd_scaled = rnd_0_to_1 - 0.5f;
            const float rnd = rnd_scaled * step_size[group];
#endif
            // Compute
            const float rnd_val = val + rnd;
            const float normalized_val_0 = rnd_val * rcp_step_size;
            const float normalized_val_1 = normalized_val_0 + 0.5f;
            const int8_t quantized_val = (int8_t) floorf(normalized_val_1);

            group_result[idx] = quantized_val;
        }

        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std <= EPS) {
            step_size[group] = group_values[0];
        }
    }
}

void quantization::DitheredQuantization8Strategy::restore(
    union Quantization_Input<int8_t>& input
) {
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float step_size = input.std_quantization_input.scale;
    const float dequantization_const = input.std_quantization_input.dequantization_const;

    const __m256 scale_vec = _mm256_set1_ps(step_size);
    const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const);
    int idx;
    for (idx = 0; idx < size - 63; idx += 64) {
        // Load
        const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (values + idx));
        const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (values + idx + 32));

        // Unpack
        __m256i unpacked_0;
        __m256i unpacked_1;
        __m256i unpacked_2;
        __m256i unpacked_3;
        __m256i unpacked_4;
        __m256i unpacked_5;
        __m256i unpacked_6;
        __m256i unpacked_7;
        Quantization8Strategy::unpack64(
            q_val_vec_0, q_val_vec_1,
            unpacked_0, unpacked_1, unpacked_2, unpacked_3,
            unpacked_4, unpacked_5, unpacked_6, unpacked_7
        );

        const __m256 f_pack_0 = _mm256_cvtepi32_ps(unpacked_0);
        const __m256 f_pack_1 = _mm256_cvtepi32_ps(unpacked_1);
        const __m256 f_pack_2 = _mm256_cvtepi32_ps(unpacked_2);
        const __m256 f_pack_3 = _mm256_cvtepi32_ps(unpacked_3);
        const __m256 f_pack_4 = _mm256_cvtepi32_ps(unpacked_4);
        const __m256 f_pack_5 = _mm256_cvtepi32_ps(unpacked_5);
        const __m256 f_pack_6 = _mm256_cvtepi32_ps(unpacked_6);
        const __m256 f_pack_7 = _mm256_cvtepi32_ps(unpacked_7);

        const __m256 f_dequantized_0 = _mm256_fmsub_ps(f_pack_0, scale_vec, dq_const_vec);
        const __m256 f_dequantized_1 = _mm256_fmsub_ps(f_pack_1, scale_vec, dq_const_vec);
        const __m256 f_dequantized_2 = _mm256_fmsub_ps(f_pack_2, scale_vec, dq_const_vec);
        const __m256 f_dequantized_3 = _mm256_fmsub_ps(f_pack_3, scale_vec, dq_const_vec);
        const __m256 f_dequantized_4 = _mm256_fmsub_ps(f_pack_4, scale_vec, dq_const_vec);
        const __m256 f_dequantized_5 = _mm256_fmsub_ps(f_pack_5, scale_vec, dq_const_vec);
        const __m256 f_dequantized_6 = _mm256_fmsub_ps(f_pack_6, scale_vec, dq_const_vec);
        const __m256 f_dequantized_7 = _mm256_fmsub_ps(f_pack_7, scale_vec, dq_const_vec);

        // Store
        float* base = result + idx;
        _mm256_storeu_ps(base + 0, f_dequantized_0);
        _mm256_storeu_ps(base + 8, f_dequantized_1);
        _mm256_storeu_ps(base + 16, f_dequantized_2);
        _mm256_storeu_ps(base + 24, f_dequantized_3);
        _mm256_storeu_ps(base + 32, f_dequantized_4);
        _mm256_storeu_ps(base + 40, f_dequantized_5);
        _mm256_storeu_ps(base + 48, f_dequantized_6);
        _mm256_storeu_ps(base + 56, f_dequantized_7);
    }

    for (; idx < size; idx++) {
        // We implicitly cast the quantized value into a float
        const float val =  values[idx];
        const float denormalized_val = step_size * val;
        result[idx] = denormalized_val - dequantization_const;
    }
}

void quantization::DitheredQuantization8Strategy::restore_parallel(union Quantization_Input<int8_t>& input) {
#if defined(_OPENMP)
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float step_size = input.std_quantization_input.scale;
    const float dequantization_const = input.std_quantization_input.dequantization_const;

    const __m256 scale_vec = _mm256_set1_ps(step_size);
    const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const);
    int idx;
    #pragma omp parallel for default(none) shared(size, values, scale_vec, dq_const_vec, result)
    for (idx = 0; idx < size - 63; idx += 64) {
        // Load
        const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (values + idx));
        const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (values + idx + 32));

        // Unpack
        __m256i unpacked_0;
        __m256i unpacked_1;
        __m256i unpacked_2;
        __m256i unpacked_3;
        __m256i unpacked_4;
        __m256i unpacked_5;
        __m256i unpacked_6;
        __m256i unpacked_7;
        Quantization8Strategy::unpack64(
            q_val_vec_0, q_val_vec_1,
            unpacked_0, unpacked_1, unpacked_2, unpacked_3,
            unpacked_4, unpacked_5, unpacked_6, unpacked_7
        );

        const __m256 f_pack_0 = _mm256_cvtepi32_ps(unpacked_0);
        const __m256 f_pack_1 = _mm256_cvtepi32_ps(unpacked_1);
        const __m256 f_pack_2 = _mm256_cvtepi32_ps(unpacked_2);
        const __m256 f_pack_3 = _mm256_cvtepi32_ps(unpacked_3);
        const __m256 f_pack_4 = _mm256_cvtepi32_ps(unpacked_4);
        const __m256 f_pack_5 = _mm256_cvtepi32_ps(unpacked_5);
        const __m256 f_pack_6 = _mm256_cvtepi32_ps(unpacked_6);
        const __m256 f_pack_7 = _mm256_cvtepi32_ps(unpacked_7);

        const __m256 f_dequantized_0 = _mm256_fmsub_ps(f_pack_0, scale_vec, dq_const_vec);
        const __m256 f_dequantized_1 = _mm256_fmsub_ps(f_pack_1, scale_vec, dq_const_vec);
        const __m256 f_dequantized_2 = _mm256_fmsub_ps(f_pack_2, scale_vec, dq_const_vec);
        const __m256 f_dequantized_3 = _mm256_fmsub_ps(f_pack_3, scale_vec, dq_const_vec);
        const __m256 f_dequantized_4 = _mm256_fmsub_ps(f_pack_4, scale_vec, dq_const_vec);
        const __m256 f_dequantized_5 = _mm256_fmsub_ps(f_pack_5, scale_vec, dq_const_vec);
        const __m256 f_dequantized_6 = _mm256_fmsub_ps(f_pack_6, scale_vec, dq_const_vec);
        const __m256 f_dequantized_7 = _mm256_fmsub_ps(f_pack_7, scale_vec, dq_const_vec);

        // Store
        float* base = result + idx;
        _mm256_storeu_ps(base + 0, f_dequantized_0);
        _mm256_storeu_ps(base + 8, f_dequantized_1);
        _mm256_storeu_ps(base + 16, f_dequantized_2);
        _mm256_storeu_ps(base + 24, f_dequantized_3);
        _mm256_storeu_ps(base + 32, f_dequantized_4);
        _mm256_storeu_ps(base + 40, f_dequantized_5);
        _mm256_storeu_ps(base + 48, f_dequantized_6);
        _mm256_storeu_ps(base + 56, f_dequantized_7);
    }

    idx = (size / 64) * 64;
    #pragma omp parallel for default(none) shared(idx, size, values, step_size, result, dequantization_const)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // We implicitly cast the quantized value into a float
        const float val =  values[idx2];
        const float denormalized_val = step_size * val;
        result[idx2] = denormalized_val - dequantization_const;
    }
#else
    quantization::DitheredQuantization8Strategy::restore(input);
#endif
}

void quantization::DitheredQuantization8Strategy::restore_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* step_size = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const int8_t* group_values = values + group_start;
        float* group_result = result + group_start;

        const __m256 scale_vec = _mm256_set1_ps(step_size[group]);
        const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const[group]);
        int idx;
        for (idx = 0; idx < group_size - 63; idx += 64) {
            // Load
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (group_values + idx));
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_values + idx + 32));

            // Unpack
            __m256i unpacked_0;
            __m256i unpacked_1;
            __m256i unpacked_2;
            __m256i unpacked_3;
            __m256i unpacked_4;
            __m256i unpacked_5;
            __m256i unpacked_6;
            __m256i unpacked_7;
            Quantization8Strategy::unpack64(
                q_val_vec_0, q_val_vec_1,
                unpacked_0, unpacked_1, unpacked_2, unpacked_3,
                unpacked_4, unpacked_5, unpacked_6, unpacked_7
            );

            const __m256 f_pack_0 = _mm256_cvtepi32_ps(unpacked_0);
            const __m256 f_pack_1 = _mm256_cvtepi32_ps(unpacked_1);
            const __m256 f_pack_2 = _mm256_cvtepi32_ps(unpacked_2);
            const __m256 f_pack_3 = _mm256_cvtepi32_ps(unpacked_3);
            const __m256 f_pack_4 = _mm256_cvtepi32_ps(unpacked_4);
            const __m256 f_pack_5 = _mm256_cvtepi32_ps(unpacked_5);
            const __m256 f_pack_6 = _mm256_cvtepi32_ps(unpacked_6);
            const __m256 f_pack_7 = _mm256_cvtepi32_ps(unpacked_7);

            const __m256 f_dequantized_0 = _mm256_fmsub_ps(f_pack_0, scale_vec, dq_const_vec);
            const __m256 f_dequantized_1 = _mm256_fmsub_ps(f_pack_1, scale_vec, dq_const_vec);
            const __m256 f_dequantized_2 = _mm256_fmsub_ps(f_pack_2, scale_vec, dq_const_vec);
            const __m256 f_dequantized_3 = _mm256_fmsub_ps(f_pack_3, scale_vec, dq_const_vec);
            const __m256 f_dequantized_4 = _mm256_fmsub_ps(f_pack_4, scale_vec, dq_const_vec);
            const __m256 f_dequantized_5 = _mm256_fmsub_ps(f_pack_5, scale_vec, dq_const_vec);
            const __m256 f_dequantized_6 = _mm256_fmsub_ps(f_pack_6, scale_vec, dq_const_vec);
            const __m256 f_dequantized_7 = _mm256_fmsub_ps(f_pack_7, scale_vec, dq_const_vec);

            // Store
            float* base = group_result + idx;
            _mm256_storeu_ps(base + 0, f_dequantized_0);
            _mm256_storeu_ps(base + 8, f_dequantized_1);
            _mm256_storeu_ps(base + 16, f_dequantized_2);
            _mm256_storeu_ps(base + 24, f_dequantized_3);
            _mm256_storeu_ps(base + 32, f_dequantized_4);
            _mm256_storeu_ps(base + 40, f_dequantized_5);
            _mm256_storeu_ps(base + 48, f_dequantized_6);
            _mm256_storeu_ps(base + 56, f_dequantized_7);
        }

        for (; idx < group_size; idx++) {
            // We implicitly cast the quantized value into a float
            const float val =  group_values[idx];
            const float denormalized_val = step_size[group] * val;
            group_result[idx] = denormalized_val - dequantization_const[group];
        }
    }
}

void quantization::DitheredQuantization8Strategy::restore_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* step_size = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, step_size, dequantization_const)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const int8_t* group_values = values + group_start;
        float* group_result = result + group_start;

        const __m256 scale_vec = _mm256_set1_ps(step_size[group]);
        const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const[group]);
        int idx;
        for (idx = 0; idx < group_size - 63; idx += 64) {
            // Load
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (group_values + idx));
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_values + idx + 32));

            // Unpack
            __m256i unpacked_0;
            __m256i unpacked_1;
            __m256i unpacked_2;
            __m256i unpacked_3;
            __m256i unpacked_4;
            __m256i unpacked_5;
            __m256i unpacked_6;
            __m256i unpacked_7;
            Quantization8Strategy::unpack64(
                q_val_vec_0, q_val_vec_1,
                unpacked_0, unpacked_1, unpacked_2, unpacked_3,
                unpacked_4, unpacked_5, unpacked_6, unpacked_7
            );

            const __m256 f_pack_0 = _mm256_cvtepi32_ps(unpacked_0);
            const __m256 f_pack_1 = _mm256_cvtepi32_ps(unpacked_1);
            const __m256 f_pack_2 = _mm256_cvtepi32_ps(unpacked_2);
            const __m256 f_pack_3 = _mm256_cvtepi32_ps(unpacked_3);
            const __m256 f_pack_4 = _mm256_cvtepi32_ps(unpacked_4);
            const __m256 f_pack_5 = _mm256_cvtepi32_ps(unpacked_5);
            const __m256 f_pack_6 = _mm256_cvtepi32_ps(unpacked_6);
            const __m256 f_pack_7 = _mm256_cvtepi32_ps(unpacked_7);

            const __m256 f_dequantized_0 = _mm256_fmsub_ps(f_pack_0, scale_vec, dq_const_vec);
            const __m256 f_dequantized_1 = _mm256_fmsub_ps(f_pack_1, scale_vec, dq_const_vec);
            const __m256 f_dequantized_2 = _mm256_fmsub_ps(f_pack_2, scale_vec, dq_const_vec);
            const __m256 f_dequantized_3 = _mm256_fmsub_ps(f_pack_3, scale_vec, dq_const_vec);
            const __m256 f_dequantized_4 = _mm256_fmsub_ps(f_pack_4, scale_vec, dq_const_vec);
            const __m256 f_dequantized_5 = _mm256_fmsub_ps(f_pack_5, scale_vec, dq_const_vec);
            const __m256 f_dequantized_6 = _mm256_fmsub_ps(f_pack_6, scale_vec, dq_const_vec);
            const __m256 f_dequantized_7 = _mm256_fmsub_ps(f_pack_7, scale_vec, dq_const_vec);

            // Store
            float* base = group_result + idx;
            _mm256_storeu_ps(base + 0, f_dequantized_0);
            _mm256_storeu_ps(base + 8, f_dequantized_1);
            _mm256_storeu_ps(base + 16, f_dequantized_2);
            _mm256_storeu_ps(base + 24, f_dequantized_3);
            _mm256_storeu_ps(base + 32, f_dequantized_4);
            _mm256_storeu_ps(base + 40, f_dequantized_5);
            _mm256_storeu_ps(base + 48, f_dequantized_6);
            _mm256_storeu_ps(base + 56, f_dequantized_7);
        }

        for (; idx < group_size; idx++) {
            // We implicitly cast the quantized value into a float
            const float val =  group_values[idx];
            const float denormalized_val = step_size[group] * val;
            group_result[idx] = denormalized_val - dequantization_const[group];
        }
    }
}