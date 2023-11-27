#include "sawb_quantization8_strategy.h"

#include <algorithm>

void quantization::SAWBQuantization8Strategy::quantize_scalar(union Quantization_Input<int8_t>& input) {
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& scale = input.std_quantization_input.scale;

    float acc_sq_mean = 0.0f;
    float acc_abs_mean = 0.0f;
    for (int idx = 0; idx < size; idx++) {
        const float val = values[idx];
        const float sq_val = val * val;
        const float abs_val = fabs(val);
        acc_sq_mean += sq_val;
        acc_abs_mean += abs_val;
    }

    const float sq_mean = acc_sq_mean / (float )size;
    const float abs_mean = acc_abs_mean / (float )size;
    const float sqrt_sq_mean = sqrt(sq_mean);
    const float clip = c1 * sqrt_sq_mean - c2 * abs_mean;
    scale = 2.0f * clip / (upperBound - lowerBound);
    for (int idx = 0; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float normalized_val = val / scale;
        const float clamped = std::min(std::max(normalized_val, lowerBound), upperBound);
        const int8_t quantized_val = (int8_t) roundf(clamped);

        // Store
        result[idx] = quantized_val;
    }
}

void quantization::SAWBQuantization8Strategy::quantize(
    union Quantization_Input<int8_t>& input
) {
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& scale = input.std_quantization_input.scale;

    __m512 acc_sq_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_sq_mean_vec_1 = _mm512_setzero_ps();
    __m512 acc_abs_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_abs_mean_vec_1 = _mm512_setzero_ps();

    // Compute clip
    int idx;
    for (idx = 0; idx < size - 31; idx += 32) {
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Calculate the absolute value
        const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
        const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

        // Calculate the sum of squares
        acc_sq_mean_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_mean_vec_0);
        acc_sq_mean_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_mean_vec_1);

        // Calculate the sum of absolute values
        acc_abs_mean_vec_0 = _mm512_add_ps(abs_val_vec_0, acc_abs_mean_vec_0);
        acc_abs_mean_vec_1 = _mm512_add_ps(abs_val_vec_1, acc_abs_mean_vec_1);
    }
    const float acc_sq_mean_0 = _mm512_haddf32_ss(acc_sq_mean_vec_0);
    const float acc_sq_mean_1 = _mm512_haddf32_ss(acc_sq_mean_vec_1);
    const float acc_abs_mean_0 = _mm512_haddf32_ss(acc_abs_mean_vec_0);
    const float acc_abs_mean_1 = _mm512_haddf32_ss(acc_abs_mean_vec_1);
    float acc_sq_mean = acc_sq_mean_0 + acc_sq_mean_1;
    float acc_abs_mean = acc_abs_mean_0 + acc_abs_mean_1;

    for (; idx < size; idx++) {
        const float val = values[idx];
        const float sq_val = val * val;
        const float abs_val = fabs(val);
        acc_sq_mean += sq_val;
        acc_abs_mean += abs_val;
    }

    // TODO: Size needs to be total size, this size is the number of non-zero elements in a sparse matrix
    const float f_rcp_size = 1.0f / (float) size;
    const float sq_mean = acc_sq_mean * f_rcp_size;
    const float abs_mean = acc_abs_mean * f_rcp_size;
    const float sqrt_sq_mean = sqrt(sq_mean);
    const float left = c1 * sqrt_sq_mean;
    const float right = c2 * abs_mean;
    const float clip = left - right;
    // TODO: Unlikely, but handle the case where the clip is 0.0f - Scale will be zero because either the input is 0 or the coefficients are selected very poorly and you are unlucky
    float clipx2; // = 2.0f * clip;
    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    if (std::abs(clip) <= EPS) {
        clipx2 = 2.0f;
    } else {
        clipx2 = 2.0f * clip;
    }
    const float bit_range = upperBound - lowerBound;
    scale = clipx2 / bit_range;
    float rcp_scale = 1.0f / scale;

    const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
    const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
    const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
    for (idx = 0; idx < size - 31; idx += 32) {
        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Normalize the values with the inverse dithered_scale
        const __m512 normalized_vec_0 = _mm512_mul_ps(val_vec_0, rcp_scale_vec);
        const __m512 normalized_vec_1 = _mm512_mul_ps(val_vec_1, rcp_scale_vec);

        // Clamp the values
        const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_vec_0, lower_bound_vec);
        const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
        const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_vec_1, lower_bound_vec);
        const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

        // Quantize the values
        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

        // Pack
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

    for (; idx < size; idx++) {
        // Load
        const float val = values[idx];

        // Compute
        const float normalized_val = val * rcp_scale;
        const float lower_clamped = normalized_val <= lowerBound ? lowerBound : normalized_val;
        const float upper_clamped = lower_clamped >= upperBound ? upperBound : lower_clamped;
        const int8_t quantized_val = (int8_t) roundf(upper_clamped);

        // Store
        result[idx] = quantized_val;
    }
}

void quantization::SAWBQuantization8Strategy::quantize_parallel(
    union Quantization_Input<int8_t>& input
) {
#if defined(_OPENMP)
    const float* values = input.std_quantization_input.dq_values;
    int8_t* result = input.std_quantization_input.q_values;
    const int size = input.std_quantization_input.size;
    float& scale = input.std_quantization_input.scale;

    const int thread_count = get_OpenMP_threads();
    __m512* acc_sq_mean_vecs_0 = new __m512[thread_count];
    __m512* acc_sq_mean_vecs_1 = new __m512[thread_count];
    __m512* acc_abs_mean_vecs_0 = new __m512[thread_count];
    __m512* acc_abs_mean_vecs_1 = new __m512[thread_count];
    for (int thread = 0; thread < thread_count; thread++) {
        acc_sq_mean_vecs_0[thread] = _mm512_setzero_ps();
        acc_sq_mean_vecs_1[thread] = _mm512_setzero_ps();
        acc_abs_mean_vecs_0[thread] = _mm512_setzero_ps();
        acc_abs_mean_vecs_1[thread] = _mm512_setzero_ps();
    }

    // Compute clip
    int idx;
    #pragma omp parallel for default(none) shared(size, values, _mm512_1st_bit_off, acc_sq_mean_vecs_0, acc_sq_mean_vecs_1, acc_abs_mean_vecs_0, acc_abs_mean_vecs_1)
    for (idx = 0; idx < size - 31; idx += 32) {
        const int tid = get_OpenMP_thread();

        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Calculate the absolute value
        const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
        const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

        // Calculate the sum of squares
        acc_sq_mean_vecs_0[tid] = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_mean_vecs_0[tid]);
        acc_sq_mean_vecs_1[tid] = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_mean_vecs_1[tid]);

        // Calculate the sum of absolute values
        acc_abs_mean_vecs_0[tid] = _mm512_add_ps(abs_val_vec_0, acc_abs_mean_vecs_0[tid]);
        acc_abs_mean_vecs_1[tid] = _mm512_add_ps(abs_val_vec_1, acc_abs_mean_vecs_1[tid]);
    }
    __m512 acc_sq_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_sq_mean_vec_1 = _mm512_setzero_ps();
    __m512 acc_abs_mean_vec_0 = _mm512_setzero_ps();
    __m512 acc_abs_mean_vec_1 = _mm512_setzero_ps();
    for (int thread = 0; thread < thread_count; thread++) {
        acc_sq_mean_vec_0 = _mm512_add_ps(acc_sq_mean_vec_0, acc_sq_mean_vecs_0[thread]);
        acc_sq_mean_vec_1 = _mm512_add_ps(acc_sq_mean_vec_1, acc_sq_mean_vecs_1[thread]);
        acc_abs_mean_vec_0 = _mm512_add_ps(acc_abs_mean_vec_0, acc_abs_mean_vecs_0[thread]);
        acc_abs_mean_vec_1 = _mm512_add_ps(acc_abs_mean_vec_1, acc_abs_mean_vecs_1[thread]);
    }
    delete [] acc_sq_mean_vecs_0;
    delete [] acc_sq_mean_vecs_1;
    delete [] acc_abs_mean_vecs_0;
    delete [] acc_abs_mean_vecs_1;

    const float acc_sq_mean_0 = _mm512_haddf32_ss(acc_sq_mean_vec_0);
    const float acc_sq_mean_1 = _mm512_haddf32_ss(acc_sq_mean_vec_1);
    const float acc_abs_mean_0 = _mm512_haddf32_ss(acc_abs_mean_vec_0);
    const float acc_abs_mean_1 = _mm512_haddf32_ss(acc_abs_mean_vec_1);
    float acc_sq_mean = acc_sq_mean_0 + acc_sq_mean_1;
    float acc_abs_mean = acc_abs_mean_0 + acc_abs_mean_1;

    idx = (size / 32) * 32;
    for (; idx < size; idx++) {
        const float val = values[idx];
        const float sq_val = val * val;
        const float abs_val = fabs(val);
        acc_sq_mean += sq_val;
        acc_abs_mean += abs_val;
    }

    // TODO: Size needs to be total size, this size is the number of non-zero elements in a sparse matrix
    const float f_rcp_size = 1.0f / (float) size;
    const float sq_mean = acc_sq_mean * f_rcp_size;
    const float abs_mean = acc_abs_mean * f_rcp_size;
    const float sqrt_sq_mean = sqrt(sq_mean);
    const float left = c1 * sqrt_sq_mean;
    const float right = c2 * abs_mean;
    const float clip = left - right;
    // TODO: Unlikely, but handle the case where the clip is 0.0f - Scale will be zero because either the input is 0 or the coefficients are selected very poorly and you are unlucky
    float clipx2; // = 2.0f * clip;
    // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
    if (std::abs(clip) <= EPS) {
        clipx2 = 2.0f;
    } else {
        clipx2 = 2.0f * clip;
    }
    const float bit_range = upperBound - lowerBound;
    scale = clipx2 / bit_range;
    float rcp_scale = 1.0f / scale;

    const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
    const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
    const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
    #pragma omp parallel for default(none) shared(size, values, rcp_scale_vec, lower_bound_vec, upper_bound_vec, result)
    for (idx = 0; idx < size - 31; idx += 32) {
        // Load
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

        // Normalize the values with the inverse dithered_scale
        const __m512 normalized_vec_0 = _mm512_mul_ps(val_vec_0, rcp_scale_vec);
        const __m512 normalized_vec_1 = _mm512_mul_ps(val_vec_1, rcp_scale_vec);

        // Clamp the values
        const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_vec_0, lower_bound_vec);
        const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
        const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_vec_1, lower_bound_vec);
        const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

        // Quantize the values
        const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
        const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

        // Pack
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

    idx = (size / 32) * 32;
    #pragma omp parallel for default(none) shared(idx, size, values, rcp_scale, result)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // Load
        const float val = values[idx2];

        // Compute
        const float normalized_val = val * rcp_scale;
        const float lower_clamped = normalized_val <= lowerBound ? lowerBound : normalized_val;
        const float upper_clamped = lower_clamped >= upperBound ? upperBound : lower_clamped;
        const int8_t quantized_val = (int8_t) roundf(upper_clamped);

        // Store
        result[idx2] = quantized_val;
    }
#else
    quantization::SAWBQuantization8Strategy::quantize(input);
#endif
}

void quantization::SAWBQuantization8Strategy::quantize_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;

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

        __m512 acc_sq_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_sq_mean_vec_1 = _mm512_setzero_ps();
        __m512 acc_abs_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_abs_mean_vec_1 = _mm512_setzero_ps();
        // Compute clip
        int idx;
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Calculate the absolute value
            const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
            const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

            // Calculate the sum of squares
            acc_sq_mean_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_mean_vec_0);
            acc_sq_mean_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_mean_vec_1);

            // Calculate the sum of absolute values
            acc_abs_mean_vec_0 = _mm512_add_ps(abs_val_vec_0, acc_abs_mean_vec_0);
            acc_abs_mean_vec_1 = _mm512_add_ps(abs_val_vec_1, acc_abs_mean_vec_1);
        }
        const float acc_sq_mean_0 = _mm512_haddf32_ss(acc_sq_mean_vec_0);
        const float acc_sq_mean_1 = _mm512_haddf32_ss(acc_sq_mean_vec_1);
        const float acc_abs_mean_0 = _mm512_haddf32_ss(acc_abs_mean_vec_0);
        const float acc_abs_mean_1 = _mm512_haddf32_ss(acc_abs_mean_vec_1);
        float acc_sq_mean = acc_sq_mean_0 + acc_sq_mean_1;
        float acc_abs_mean = acc_abs_mean_0 + acc_abs_mean_1;

        for (; idx < group_size; idx++) {
            const float val = group_values[idx];
            const float sq_val = val * val;
            const float abs_val = fabs(val);
            acc_sq_mean += sq_val;
            acc_abs_mean += abs_val;
        }

        const float f_rcp_size = 1.0f / (float) group_size;
        const float sq_mean = acc_sq_mean * f_rcp_size;
        const float abs_mean = acc_abs_mean * f_rcp_size;
        const float sqrt_sq_mean = sqrt(sq_mean);
        const float left = c1 * sqrt_sq_mean;
        const float right = c2 * abs_mean;
        const float clip = left - right;
        float clipx2; // = 2.0f * clip;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std::abs(clip) <= EPS) {
            clipx2 = 2.0f;
        } else {
            clipx2 = 2.0f * clip;
        }
        const float bit_range = upperBound - lowerBound;
        scale[group] = clipx2 / bit_range;
        float rcp_scale = 1.0f / scale[group];

        const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
        const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
        const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Normalize the values with the inverse dithered_scale
            const __m512 normalized_vec_0 = _mm512_mul_ps(val_vec_0, rcp_scale_vec);
            const __m512 normalized_vec_1 = _mm512_mul_ps(val_vec_1, rcp_scale_vec);

            // Clamp the values
            const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_vec_0, lower_bound_vec);
            const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
            const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_vec_1, lower_bound_vec);
            const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

            // Quantize the values
            const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

            // Pack
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

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float normalized_val = val * rcp_scale;
            const float lower_clamped = normalized_val <= lowerBound ? lowerBound : normalized_val;
            const float upper_clamped = lower_clamped >= upperBound ? upperBound : lower_clamped;
            const int8_t quantized_val = (int8_t) roundf(upper_clamped);

            // Store
            group_result[idx] = quantized_val;
        }
    }
}

void quantization::SAWBQuantization8Strategy::quantize_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.std_grouped_input.dq_values;
    int8_t* result = input.std_grouped_input.q_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, _mm512_1st_bit_off, EPS, scale)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const float* group_values = values + group_start;
        int8_t* group_result = result + group_start;

        __m512 acc_sq_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_sq_mean_vec_1 = _mm512_setzero_ps();
        __m512 acc_abs_mean_vec_0 = _mm512_setzero_ps();
        __m512 acc_abs_mean_vec_1 = _mm512_setzero_ps();
        // Compute clip
        int idx;
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Calculate the absolute value
            const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
            const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

            // Calculate the sum of squares
            acc_sq_mean_vec_0 = _mm512_fmadd_ps(val_vec_0, val_vec_0, acc_sq_mean_vec_0);
            acc_sq_mean_vec_1 = _mm512_fmadd_ps(val_vec_1, val_vec_1, acc_sq_mean_vec_1);

            // Calculate the sum of absolute values
            acc_abs_mean_vec_0 = _mm512_add_ps(abs_val_vec_0, acc_abs_mean_vec_0);
            acc_abs_mean_vec_1 = _mm512_add_ps(abs_val_vec_1, acc_abs_mean_vec_1);
        }
        const float acc_sq_mean_0 = _mm512_haddf32_ss(acc_sq_mean_vec_0);
        const float acc_sq_mean_1 = _mm512_haddf32_ss(acc_sq_mean_vec_1);
        const float acc_abs_mean_0 = _mm512_haddf32_ss(acc_abs_mean_vec_0);
        const float acc_abs_mean_1 = _mm512_haddf32_ss(acc_abs_mean_vec_1);
        float acc_sq_mean = acc_sq_mean_0 + acc_sq_mean_1;
        float acc_abs_mean = acc_abs_mean_0 + acc_abs_mean_1;

        for (; idx < group_size; idx++) {
            const float val = group_values[idx];
            const float sq_val = val * val;
            const float abs_val = fabs(val);
            acc_sq_mean += sq_val;
            acc_abs_mean += abs_val;
        }

        const float f_rcp_size = 1.0f / (float) group_size;
        const float sq_mean = acc_sq_mean * f_rcp_size;
        const float abs_mean = acc_abs_mean * f_rcp_size;
        const float sqrt_sq_mean = sqrt(sq_mean);
        const float left = c1 * sqrt_sq_mean;
        const float right = c2 * abs_mean;
        const float clip = left - right;
        float clipx2; // = 2.0f * clip;
        // TODO: Do float comparisons better, USE CONSTANTS INSTEAD OF SOME MAGIC NUMBER
        if (std::abs(clip) <= EPS) {
            clipx2 = 2.0f;
        } else {
            clipx2 = 2.0f * clip;
        }
        const float bit_range = upperBound - lowerBound;
        scale[group] = clipx2 / bit_range;
        float rcp_scale = 1.0f / scale[group];

        const __m512 rcp_scale_vec = _mm512_set1_ps(rcp_scale);
        const __m512 lower_bound_vec = _mm512_set1_ps(lowerBound);
        const __m512 upper_bound_vec = _mm512_set1_ps(upperBound);
        idx = 0;
        for (; idx < group_size - 31; idx += 32) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

            // Normalize the values with the inverse dithered_scale
            const __m512 normalized_vec_0 = _mm512_mul_ps(val_vec_0, rcp_scale_vec);
            const __m512 normalized_vec_1 = _mm512_mul_ps(val_vec_1, rcp_scale_vec);

            // Clamp the values
            const __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_vec_0, lower_bound_vec);
            const __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
            const __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_vec_1, lower_bound_vec);
            const __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

            // Quantize the values
            const __m512i quantized_vec_0 = _mm512_cvtps_epi32(upper_clamped_vec_0);
            const __m512i quantized_vec_1 = _mm512_cvtps_epi32(upper_clamped_vec_1);

            // Pack
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

        for (; idx < group_size; idx++) {
            // Load
            const float val = group_values[idx];

            // Compute
            const float normalized_val = val * rcp_scale;
            const float lower_clamped = normalized_val <= lowerBound ? lowerBound : normalized_val;
            const float upper_clamped = lower_clamped >= upperBound ? upperBound : lower_clamped;
            const int8_t quantized_val = (int8_t) roundf(upper_clamped);

            // Store
            group_result[idx] = quantized_val;
        }
    }
}

void quantization::SAWBQuantization8Strategy::restore(
    union Quantization_Input<int8_t>& input
) {
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float scale = input.std_quantization_input.scale;

    const __m256 scale_vec = _mm256_set1_ps(scale);
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

        const __m256 f_dequantized_0 = _mm256_mul_ps(f_pack_0, scale_vec);
        const __m256 f_dequantized_1 = _mm256_mul_ps(f_pack_1, scale_vec);
        const __m256 f_dequantized_2 = _mm256_mul_ps(f_pack_2, scale_vec);
        const __m256 f_dequantized_3 = _mm256_mul_ps(f_pack_3, scale_vec);
        const __m256 f_dequantized_4 = _mm256_mul_ps(f_pack_4, scale_vec);
        const __m256 f_dequantized_5 = _mm256_mul_ps(f_pack_5, scale_vec);
        const __m256 f_dequantized_6 = _mm256_mul_ps(f_pack_6, scale_vec);
        const __m256 f_dequantized_7 = _mm256_mul_ps(f_pack_7, scale_vec);

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
        // Load
        const float val = values[idx];

        // Compute + Store
        result[idx] = val * scale;
    }
}

void quantization::SAWBQuantization8Strategy::restore_parallel(union Quantization_Input<int8_t>& input) {
#if defined(_OPENMP)
    const int8_t* values = input.std_quantization_input.q_values;
    float* result = input.std_quantization_input.dq_values;
    const int size = input.std_quantization_input.size;
    const float scale = input.std_quantization_input.scale;

    const __m256 scale_vec = _mm256_set1_ps(scale);
    int idx;
    #pragma omp parallel for default(none) shared(size, values, scale_vec, result)
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

        const __m256 f_dequantized_0 = _mm256_mul_ps(f_pack_0, scale_vec);
        const __m256 f_dequantized_1 = _mm256_mul_ps(f_pack_1, scale_vec);
        const __m256 f_dequantized_2 = _mm256_mul_ps(f_pack_2, scale_vec);
        const __m256 f_dequantized_3 = _mm256_mul_ps(f_pack_3, scale_vec);
        const __m256 f_dequantized_4 = _mm256_mul_ps(f_pack_4, scale_vec);
        const __m256 f_dequantized_5 = _mm256_mul_ps(f_pack_5, scale_vec);
        const __m256 f_dequantized_6 = _mm256_mul_ps(f_pack_6, scale_vec);
        const __m256 f_dequantized_7 = _mm256_mul_ps(f_pack_7, scale_vec);

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
    #pragma omp parallel for default(none) shared(idx, size, values, result, scale)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // Load
        const float val = values[idx2];

        // Compute + Store
        result[idx2] = val * scale;
    }
#else
    quantization::SAWBQuantization8Strategy::restore(input);
#endif
}

void quantization::SAWBQuantization8Strategy::restore_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;

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

            const __m256 f_dequantized_0 = _mm256_mul_ps(f_pack_0, scale_vec);
            const __m256 f_dequantized_1 = _mm256_mul_ps(f_pack_1, scale_vec);
            const __m256 f_dequantized_2 = _mm256_mul_ps(f_pack_2, scale_vec);
            const __m256 f_dequantized_3 = _mm256_mul_ps(f_pack_3, scale_vec);
            const __m256 f_dequantized_4 = _mm256_mul_ps(f_pack_4, scale_vec);
            const __m256 f_dequantized_5 = _mm256_mul_ps(f_pack_5, scale_vec);
            const __m256 f_dequantized_6 = _mm256_mul_ps(f_pack_6, scale_vec);
            const __m256 f_dequantized_7 = _mm256_mul_ps(f_pack_7, scale_vec);

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
            // Load
            const float val = group_values[idx];

            // Compute + Store
            group_result[idx] = val * scale[group];
        }
    }
}

void quantization::SAWBQuantization8Strategy::restore_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.std_grouped_input.q_values;
    float* result = input.std_grouped_input.dq_values;
    const int size = input.std_grouped_input.size;
    float* scale = input.std_grouped_input.scale;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, scale)
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

            const __m256 f_dequantized_0 = _mm256_mul_ps(f_pack_0, scale_vec);
            const __m256 f_dequantized_1 = _mm256_mul_ps(f_pack_1, scale_vec);
            const __m256 f_dequantized_2 = _mm256_mul_ps(f_pack_2, scale_vec);
            const __m256 f_dequantized_3 = _mm256_mul_ps(f_pack_3, scale_vec);
            const __m256 f_dequantized_4 = _mm256_mul_ps(f_pack_4, scale_vec);
            const __m256 f_dequantized_5 = _mm256_mul_ps(f_pack_5, scale_vec);
            const __m256 f_dequantized_6 = _mm256_mul_ps(f_pack_6, scale_vec);
            const __m256 f_dequantized_7 = _mm256_mul_ps(f_pack_7, scale_vec);

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
            // Load
            const float val = group_values[idx];

            // Compute + Store
            group_result[idx] = val * scale[group];
        }
    }
}