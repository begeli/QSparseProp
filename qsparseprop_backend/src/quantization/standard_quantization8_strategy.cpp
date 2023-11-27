#include "standard_quantization8_strategy.h"
#include <iostream>

// TODO: This code needs to return dithered_scale somehow
void quantization::StandardQuantization8Strategy::quantize(
    union Quantization_Input<int8_t>& input
) {
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& scale = input.std_quantization_input.scale;
    float& dequantization_const = input.std_quantization_input.dequantization_const;

    // Find the minimum and maximum values to compute the scale_vec
    __m512 min_vec_0 = _mm512_set1_ps(MAX_FLOAT);
    __m512 max_vec = _mm512_set1_ps(MIN_FLOAT);

    int idx;
    // TODO: Implement loop unrolling
    for (idx = 0; idx < size - 15; idx += 16) {
        // Load - Unaligned load is fine, it has the same latency as aligned load, so there is no need for peeling.
        const __m512 candidates = _mm512_loadu_ps(values + idx); // throughput 0.5 CPI

        // Compute
        const __m512 candidate_max_vec = _mm512_max_ps(candidates, max_vec); // throughput 1 CPI
        const __m512 candidate_min_vec = _mm512_min_ps(candidates, min_vec_0);

        // "Store"
        min_vec_0 = candidate_min_vec;
        max_vec = candidate_max_vec;
    }
    float min = _mm512_hminf32_ss(min_vec_0);
    float max = _mm512_hmaxf32_ss(max_vec);

    // Handle the remaining elements
    for (; idx < size; idx++) {
        // Load
        const float value = values[idx];

        // Compute
        const float candidate_max = max >= value ? max : value;
        const float candidate_min = min <= value ? min : value;

        // "Store"
        max = candidate_max;
        min = candidate_min;
    }

    // Used during clean up at the end
    const float value_range = max - min;
    const float bit_range = upperBound - lowerBound;
    float rcp_scale;
    // Avoid div by 0 error.
    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    if (value_range < EPS) {
        // Quantizes everything to lower bound
        scale = 1.0f; // max / lowerBound;
        rcp_scale = 1.0f;
    } else {
        scale = value_range / bit_range;
        rcp_scale = 1.0f / scale;
    }
    float quantization_multiplier_0 = lowerBound * scale;
    dequantization_const = quantization_multiplier_0 - min;
    const float quantization_multiplier_1 = rcp_scale * dequantization_const;

    const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
    const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
    const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
    const __m512 quantization_multiplier_vec = _mm512_set1_ps(quantization_multiplier_1);
    // TODO: Implement loop unrolling
    for (idx = 0; idx < size - 31; idx += 32) {
        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Compute
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const __m512 rnd_0 = _mm512_setzero_ps();
        const __m512 rnd_1 = _mm512_setzero_ps();
#else
        const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

        const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
        const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

        const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
        const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

        const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
        const __m512 normalized_vec_0 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_0, quantization_multiplier_vec);
        const __m512 normalized_vec_1 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_1, quantization_multiplier_vec);
        const __m512 normalized_rnd_vec_0 = _mm512_add_ps(normalized_vec_0, rnd_0);
        const __m512 normalized_rnd_vec_1 = _mm512_add_ps(normalized_vec_1, rnd_1);

        // Clamp the values out of bounds
        const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_rnd_vec_0, lower_bound_vec);
        const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
        const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_rnd_vec_1, lower_bound_vec);
        const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

        // Quantize the values
        // TODO: Use rounding instead of truncation ffs...
        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

        // Store
        // Step 1: break the 512bit vectors into two
        const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
        const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
        const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
        const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

        // Step 2: Pack the vectors into a single vector
        __m256i pack8;
        Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

        // Step 3: Store the combined vectors in the result array
        _mm256_storeu_si256((__m256i *) (result + idx), pack8);
    }

    // Handle the remaining elements
    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float normalized_val_0 = rcp_scale * val;
        const float normalized_val_1 = normalized_val_0 + quantization_multiplier_1;
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const float rnd = 0;
#else
        const float rnd_0_to_1 = get_random_float();
        const float rnd = rnd_0_to_1 - 0.5f;
#endif
        const float normalized_rnd_val_0 = normalized_val_1 + rnd;
        const float lower_clamped = normalized_rnd_val_0 >= lowerBound ? normalized_rnd_val_0 : lowerBound;
        const float upper_clamped = lower_clamped <= upperBound ? lower_clamped : upperBound;
        // TODO: Use rounding instead of truncation ffs... roundf(float)
        const int8_t rounded_down = (int8_t) floorf(upper_clamped);

        // Store
        result[idx] = rounded_down;
    }
}

void quantization::StandardQuantization8Strategy::quantize_parallel(
    union Quantization_Input<int8_t>& input
) {
#if defined(_OPENMP)
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& scale = input.std_quantization_input.scale;
    float& dequantization_const = input.std_quantization_input.dequantization_const;

    // Find the minimum and maximum values to compute the scale_vec
    const int thread_count = get_OpenMP_threads();
    __m512* min_vecs = new __m512[thread_count];
    __m512* max_vecs = new __m512[thread_count];
    for (int thread = 0; thread < thread_count; thread++) {
        min_vecs[thread] = _mm512_set1_ps(MAX_FLOAT);
        max_vecs[thread] = _mm512_set1_ps(MIN_FLOAT);
    }

    int idx;
    // TODO: Implement loop unrolling
    #pragma omp parallel for default(none) shared(size, values, min_vecs, max_vecs)
    for (idx = 0; idx < size - 15; idx += 16) {
        const int tid = get_OpenMP_thread();
        // Load - Unaligned load is fine, it has the same latency as aligned load, so there is no need for peeling.
        const __m512 candidates = _mm512_loadu_ps(values + idx); // throughput 0.5 CPI

        // Compute
        const __m512 candidate_max_vec = _mm512_max_ps(candidates, max_vecs[tid]); // throughput 1 CPI
        const __m512 candidate_min_vec = _mm512_min_ps(candidates, min_vecs[tid]);

        // "Store"
        min_vecs[tid] = candidate_min_vec;
        max_vecs[tid] = candidate_max_vec;
    }

    __m512 min_vec_0 = _mm512_set1_ps(MAX_FLOAT);
    __m512 max_vec = _mm512_set1_ps(MIN_FLOAT);
    for (int i = 0; i < thread_count; i++) {
        min_vec_0 = _mm512_min_ps(min_vec_0, min_vecs[i]);
        max_vec = _mm512_max_ps(max_vec, max_vecs[i]);
    }
    delete [] min_vecs;
    delete [] max_vecs;
    float min = _mm512_hminf32_ss(min_vec_0);
    float max = _mm512_hmaxf32_ss(max_vec);

    // Handle the remaining elements
    idx = (size / 16) * 16;
    for (; idx < size; idx++) {
        // Load
        const float value = values[idx];

        // Compute
        const float candidate_max = max >= value ? max : value;
        const float candidate_min = min <= value ? min : value;

        // "Store"
        max = candidate_max;
        min = candidate_min;
    }

    // Used during clean up at the end
    const float value_range = max - min;
    const float bit_range = upperBound - lowerBound;
    float rcp_scale;
    // Avoid div by 0 error.
    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    if (value_range < EPS) {
        // Quantizes everything to lower bound
        scale = 1.0f; // max / lowerBound;
        rcp_scale = 1.0f;
    } else {
        scale = value_range / bit_range;
        rcp_scale = 1.0f / scale;
    }
    float quantization_multiplier_0 = lowerBound * scale;
    dequantization_const = quantization_multiplier_0 - min;
    const float quantization_multiplier_1 = rcp_scale * dequantization_const;

    const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
    const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
    const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
    const __m512 quantization_multiplier_vec = _mm512_set1_ps(quantization_multiplier_1);
    // TODO: Implement loop unrolling
    #pragma omp parallel for default(none) shared(size, values, rcp_scale_vec, quantization_multiplier_vec, lower_bound_vec, upper_bound_vec, result, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, _mm512_0_5_ps, avx512_random_key1_perthread, avx512_random_key2_perthread)
    for (idx = 0; idx < size - 31; idx += 32) {
        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Compute
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

        const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
        const __m512 normalized_vec_0 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_0, quantization_multiplier_vec);
        const __m512 normalized_vec_1 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_1, quantization_multiplier_vec);
        const __m512 normalized_rnd_vec_0 = _mm512_add_ps(normalized_vec_0, rnd_0);
        const __m512 normalized_rnd_vec_1 = _mm512_add_ps(normalized_vec_1, rnd_1);

        // Clamp the values out of bounds
        const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_rnd_vec_0, lower_bound_vec);
        const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
        const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_rnd_vec_1, lower_bound_vec);
        const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

        // Quantize the values
        // TODO: Use rounding instead of truncation ffs...
        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

        // Store
        // Step 1: break the 512bit vectors into two
        const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
        const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
        const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
        const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

        // Step 2: Pack the vectors into a single vector
        __m256i pack8;
        Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

        // Step 3: Store the combined vectors in the result array
        _mm256_storeu_si256((__m256i *) (result + idx), pack8);
    }


    // Handle the remaining elements
    idx = (size / 32) * 32; // 32 looks like a magic number... It is the loop unroll amount in the above loop.
    #pragma omp parallel for default(none) shared(idx, size, values, rcp_scale, quantization_multiplier_1, result)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // Load
        const float val = values[idx2];

        // Compute
        const float normalized_val_0 = rcp_scale * val;
        const float normalized_val_1 = normalized_val_0 + quantization_multiplier_1;
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const float rnd = 0;
#else
        const float rnd_0_to_1 = get_random_float();
        const float rnd = rnd_0_to_1 - 0.5f;
#endif
        const float normalized_rnd_val_0 = normalized_val_1 + rnd;
        const float lower_clamped = normalized_rnd_val_0 >= lowerBound ? normalized_rnd_val_0 : lowerBound;
        const float upper_clamped = lower_clamped <= upperBound ? lower_clamped : upperBound;
        // TODO: Use rounding instead of truncation ffs... roundf(float)
        const int8_t rounded_down = (int8_t) floorf(upper_clamped);

        // Store
        result[idx2] = rounded_down;
    }
#else
    quantization::StandardQuantization8Strategy::quantize(input);
#endif
}

void quantization::StandardQuantization8Strategy::quantize_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;
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

        // Find the minimum and maximum values to compute the scale_vec
        __m512 min_vec_0 = _mm512_set1_ps(MAX_FLOAT);
        __m512 max_vec = _mm512_set1_ps(MIN_FLOAT);
        int idx;
        for (idx = 0; idx < group_size - 15; idx += 16) {
            // Load - Unaligned load is fine, it has the same latency as aligned load, so there is no need for peeling.
            const __m512 candidates = _mm512_loadu_ps(group_values + idx); // throughput 0.5 CPI

            // Compute
            const __m512 candidate_max_vec = _mm512_max_ps(candidates, max_vec); // throughput 1 CPI
            const __m512 candidate_min_vec = _mm512_min_ps(candidates, min_vec_0);

            // "Store"
            min_vec_0 = candidate_min_vec;
            max_vec = candidate_max_vec;
        }
        float min = _mm512_hminf32_ss(min_vec_0);
        float max = _mm512_hmaxf32_ss(max_vec);

        // Handle the remaining elements
        for (; idx < group_size; idx++) {
            // Load
            const float value = group_values[idx];

            // Compute
            const float candidate_max = max >= value ? max : value;
            const float candidate_min = min <= value ? min : value;

            // "Store"
            max = candidate_max;
            min = candidate_min;
        }

        // Used during clean up at the end
        const float value_range = max - min;
        const float bit_range = upperBound - lowerBound;
        float rcp_scale;
        // Avoid div by 0 error.
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (value_range < EPS) {
            // Quantizes everything to lower bound
            scale[group] = 1.0f; // max / lowerBound;
            rcp_scale = 1.0f;
        } else {
            scale[group] = value_range / bit_range;
            rcp_scale = 1.0f / scale[group];
        }
        float quantization_multiplier_0 = lowerBound * scale[group];
        dequantization_const[group] = quantization_multiplier_0 - min;
        const float quantization_multiplier_1 = rcp_scale * dequantization_const[group];

        const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
        const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
        const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
        const __m512 quantization_multiplier_vec = _mm512_set1_ps(quantization_multiplier_1);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Compute
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const __m512 rnd_0 = _mm512_setzero_ps();
            const __m512 rnd_1 = _mm512_setzero_ps();
#else
            const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

            const __m512i rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
            const __m512i rnd_i8_1 = _mm512_slli_epi32(rnd_i8_0,  8);

            const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
            const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

            const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
            const __m512 normalized_vec_0 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_0, quantization_multiplier_vec);
            const __m512 normalized_vec_1 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_1, quantization_multiplier_vec);
            const __m512 normalized_rnd_vec_0 = _mm512_add_ps(normalized_vec_0, rnd_0);
            const __m512 normalized_rnd_vec_1 = _mm512_add_ps(normalized_vec_1, rnd_1);

            // Clamp the values out of bounds
            const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_rnd_vec_0, lower_bound_vec);
            const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
            const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_rnd_vec_1, lower_bound_vec);
            const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

            // Quantize the values
            // TODO: Use rounding instead of truncation ffs...
            const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

            // Store
            // Step 1: break the 512bit vectors into two
            const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
            const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
            const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
            const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

            // Step 2: Pack the vectors into a single vector
            __m256i pack8;
            Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

            // Step 3: Store the combined vectors in the result array
            _mm256_storeu_si256((__m256i *) (group_result + idx), pack8);
        }

        // Handle the remaining elements
        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float normalized_val_0 = rcp_scale * val;
            const float normalized_val_1 = normalized_val_0 + quantization_multiplier_1;
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float rnd = 0;
#else
            const float rnd_0_to_1 = get_random_float();
            const float rnd = rnd_0_to_1 - 0.5f;
#endif
            const float normalized_rnd_val_0 = normalized_val_1 + rnd;
            const float lower_clamped = normalized_rnd_val_0 >= lowerBound ? normalized_rnd_val_0 : lowerBound;
            const float upper_clamped = lower_clamped <= upperBound ? lower_clamped : upperBound;
            // TODO: Use rounding instead of truncation ffs... roundf(float)
            const int8_t rounded_down = (int8_t) floorf(upper_clamped);

            // Store
            group_result[idx] = rounded_down;
        }
    }
}

void quantization::StandardQuantization8Strategy::quantize_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, MAX_FLOAT, MIN_FLOAT, EPS, scale, dequantization_const, avx512_random_key1_perthread, avx512_random_key2_perthread, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, _mm512_0_5_ps)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const float* group_values = values + group_start;
        int8_t* group_result = result + group_start;

        // Find the minimum and maximum values to compute the scale_vec
        __m512 min_vec_0 = _mm512_set1_ps(MAX_FLOAT);
        __m512 max_vec = _mm512_set1_ps(MIN_FLOAT);
        int idx;
        for (idx = 0; idx < group_size - 15; idx += 16) {
            // Load - Unaligned load is fine, it has the same latency as aligned load, so there is no need for peeling.
            const __m512 candidates = _mm512_loadu_ps(group_values + idx); // throughput 0.5 CPI

            // Compute
            const __m512 candidate_max_vec = _mm512_max_ps(candidates, max_vec); // throughput 1 CPI
            const __m512 candidate_min_vec = _mm512_min_ps(candidates, min_vec_0);

            // "Store"
            min_vec_0 = candidate_min_vec;
            max_vec = candidate_max_vec;
        }
        float min = _mm512_hminf32_ss(min_vec_0);
        float max = _mm512_hmaxf32_ss(max_vec);

        // Handle the remaining elements
        for (; idx < group_size; idx++) {
            // Load
            const float value = group_values[idx];

            // Compute
            const float candidate_max = max >= value ? max : value;
            const float candidate_min = min <= value ? min : value;

            // "Store"
            max = candidate_max;
            min = candidate_min;
        }

        // Used during clean up at the end
        const float value_range = max - min;
        const float bit_range = upperBound - lowerBound;
        float rcp_scale;
        // Avoid div by 0 error.
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (value_range < EPS) {
            // Quantizes everything to lower bound
            scale[group] = 1.0f; // max / lowerBound;
            rcp_scale = 1.0f;
        } else {
            scale[group] = value_range / bit_range;
            rcp_scale = 1.0f / scale[group];
        }
        float quantization_multiplier_0 = lowerBound * scale[group];
        dequantization_const[group] = quantization_multiplier_0 - min;
        const float quantization_multiplier_1 = rcp_scale * dequantization_const[group];

        const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
        const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
        const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
        const __m512 quantization_multiplier_vec = _mm512_set1_ps(quantization_multiplier_1);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Compute
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

            const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
            const __m512 normalized_vec_0 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_0, quantization_multiplier_vec);
            const __m512 normalized_vec_1 = _mm512_fmadd_ps(rcp_scale_vec, val_vec_1, quantization_multiplier_vec);
            const __m512 normalized_rnd_vec_0 = _mm512_add_ps(normalized_vec_0, rnd_0);
            const __m512 normalized_rnd_vec_1 = _mm512_add_ps(normalized_vec_1, rnd_1);

            // Clamp the values out of bounds
            const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_rnd_vec_0, lower_bound_vec);
            const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
            const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_rnd_vec_1, lower_bound_vec);
            const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

            // Quantize the values
            // TODO: Use rounding instead of truncation ffs...
            const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

            // Store
            // Step 1: break the 512bit vectors into two
            const __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0); // Cast has 0 latency!!!
            const __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
            const __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
            const __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

            // Step 2: Pack the vectors into a single vector
            __m256i pack8;
            Quantization8Strategy::pack32(q_vec_0_lo, q_vec_0_hi, q_vec_1_lo, q_vec_1_hi, pack8);

            // Step 3: Store the combined vectors in the result array
            _mm256_storeu_si256((__m256i *) (group_result + idx), pack8);
        }

        // Handle the remaining elements
        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float normalized_val_0 = rcp_scale * val;
            const float normalized_val_1 = normalized_val_0 + quantization_multiplier_1;
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float rnd = 0;
#else
            const float rnd_0_to_1 = get_random_float();
            const float rnd = rnd_0_to_1 - 0.5f;
#endif
            const float normalized_rnd_val_0 = normalized_val_1 + rnd;
            const float lower_clamped = normalized_rnd_val_0 >= lowerBound ? normalized_rnd_val_0 : lowerBound;
            const float upper_clamped = lower_clamped <= upperBound ? lower_clamped : upperBound;
            // TODO: Use rounding instead of truncation ffs... roundf(float)
            const int8_t rounded_down = (int8_t) floorf(upper_clamped);

            // Store
            group_result[idx] = rounded_down;
        }
    }
}

void quantization::StandardQuantization8Strategy::restore(union Quantization_Input<int8_t>& input) {
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float scale = input.std_quantization_input.scale;
    const float dequantization_const = input.std_quantization_input.dequantization_const;

    const __m256 scale_vec = _mm256_set1_ps(scale);
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
        result[idx] = scale * val - dequantization_const;
    }
}

void quantization::StandardQuantization8Strategy::restore_parallel(union Quantization_Input<int8_t>& input) {
#if defined(_OPENMP)
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float scale = input.std_quantization_input.scale;
    const float dequantization_const = input.std_quantization_input.dequantization_const;

    const __m256 scale_vec = _mm256_set1_ps(scale);
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
    #pragma omp parallel for default(none) shared(size, values, result, scale, dequantization_const, idx)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // We implicitly cast the quantized value into a float
        const float val =  values[idx2];
        result[idx2] = scale * val - dequantization_const;
    }
#else
    quantization::StandardQuantization8Strategy::restore(input);
#endif
}

void quantization::StandardQuantization8Strategy::restore_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;
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

        const __m256 scale_vec = _mm256_set1_ps(scale[group]);
        const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const[group]);
        int idx;
        for (idx = 0; idx < group_size - 63; idx += 64) {
            // Load
            const int8_t* group_base = group_values + idx;
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) group_base);
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_base + 32));

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
            group_result[idx] = scale[group] * val - dequantization_const[group];
        }
    }
}

void quantization::StandardQuantization8Strategy::restore_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;
    float* dequantization_const = input.std_grouped_input.dequantization_const;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, scale, dequantization_const)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const int8_t* group_values = values + group_start;
        float* group_result = result + group_start;

        const __m256 scale_vec = _mm256_set1_ps(scale[group]);
        const __m256 dq_const_vec = _mm256_set1_ps(dequantization_const[group]);
        int idx;
        for (idx = 0; idx < group_size - 63; idx += 64) {
            // Load
            const int8_t* group_base = group_values + idx;
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) group_base);
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_base + 32));

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
            group_result[idx] = scale[group] * val - dequantization_const[group];
        }
    }
}
