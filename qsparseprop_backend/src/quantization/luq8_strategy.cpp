#include "luq8_strategy.h"

// IMPORTANT NOTE: THIS CODE REQUIRES -fno-finite-math-only FLAG TO WORK CORRECTLY, O.W. WE
// CAN'T CHECK FOR INFINITIES

void quantization::LUQ8Strategy::quantize(
    union Quantization_Input<int8_t>& input
) {
    const float* values = input.luq_quantization_input.dq_values;
    int8_t* result = input.luq_quantization_input.q_values;
    const int size = input.luq_quantization_input.size;
    const int mask_count = input.luq_quantization_input.mask_count;
    __mmask16* signs = input.luq_quantization_input.signs;

    __m512 acc_max_0 = _mm512_set1_ps(MIN_FLOAT);
    __m512 acc_max_1 = _mm512_set1_ps(MIN_FLOAT);

    // Extract signs and find max(abs(values))
    int idx;
    int signIdx;
    for (idx = 0, signIdx = 0; idx < size - 31; idx += 32, signIdx += 2) {
        // Load
        const __m512 val_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_1 = _mm512_loadu_ps(values + idx + 16);

        // Compute
        const __mmask16 neg_mask_0 = _mm512_cmp_ps_mask(val_0, _mm512_zeros_ps, _CMP_LT_OQ);
        const __mmask16 neg_mask_1 = _mm512_cmp_ps_mask(val_1, _mm512_zeros_ps, _CMP_LT_OQ);
        const __m512 abs_val_0 = _mm512_and_ps(val_0, _mm512_1st_bit_off);
        const __m512 abs_val_1 = _mm512_and_ps(val_1, _mm512_1st_bit_off);
        const __m512 max_vec_0 = _mm512_max_ps(abs_val_0, acc_max_0);
        const __m512 max_vec_1 = _mm512_max_ps(abs_val_1, acc_max_1);

        // Store
        acc_max_0 = max_vec_0;
        acc_max_1 = max_vec_1;
        signs[signIdx] = neg_mask_0;
        signs[signIdx + 1] = neg_mask_1;
    }
    const float max_0 = _mm512_hmaxf32_ss(acc_max_0);
    const float max_1 = _mm512_hmaxf32_ss(acc_max_1);
    float max = max_0 <= max_1 ? max_1 : max_0;

    uint32_t mask = 0;
    int bitIdx;
    for (bitIdx = 0; idx < size; idx++, bitIdx++) {
        // Load
        const float val = fabs(values[idx]);

        // Compute
        const float curr_max = max <= val ? val : max;
        const uint32_t bit = values[idx] < 0 ? (1 << bitIdx) : 0; // TODO: Try using (values[idx] < 0) << bitIdx

        // Store
        max = curr_max;
        mask = mask | bit;
    }
    if (signIdx < mask_count) {
        signs[signIdx++] = mask;
    }
    if (signIdx < mask_count) {
        signs[signIdx] = mask >> 16;
    }
    const float alpha = max * threshold_denom;
    const float rcp_alpha = 1.0f / alpha;
    input.luq_quantization_input.scale = alpha;

    const __m512 alpha_vec = _mm512_set1_ps(alpha);
    const __m512 rcp_alpha_vec = _mm512_set1_ps(rcp_alpha);
    for (idx = 0; idx < size - 31; idx += 32) {
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const __m512 alpha_rnd_0 = _mm512_set1_ps(1.0f);
        const __m512 alpha_rnd_1 = _mm512_set1_ps(1.0f);
        const __m512 rnd_0 = _mm512_setzero_ps();
        const __m512 rnd_1 = _mm512_setzero_ps();
#else
        const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

        const __m512i alpha_rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
        const __m512i alpha_rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 8);
        const __m512i rnd_i8_0 = _mm512_slli_epi32(alpha_rnd_i8_0, 16);
        const __m512i rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 24);

        const __m512 alpha_rnd_f8_0 = _mm512_cvtepi32_ps(alpha_rnd_i8_0);
        const __m512 alpha_rnd_f8_1 = _mm512_cvtepi32_ps(alpha_rnd_i8_1);
        const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
        const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

        const __m512 alpha_rnd_0 = _mm512_mul_ps(alpha_rnd_f8_0, _mm512_rcp_2pow31_ps);
        const __m512 alpha_rnd_1 = _mm512_mul_ps(alpha_rnd_f8_1, _mm512_rcp_2pow31_ps);
        const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
        const __m512 alpha_eps_vec_0 = _mm512_mul_ps(alpha_vec, alpha_rnd_0);
        const __m512 alpha_eps_vec_1 = _mm512_mul_ps(alpha_vec, alpha_rnd_1);

        const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
        const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

        const __m512 sign_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_on);
        const __m512 sign_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_on);

        const __mmask16 abs_lt_alpha_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_vec, _CMP_LT_OQ);
        const __mmask16 abs_lt_alpha_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_vec, _CMP_LT_OQ);
        const __m512 alpha_mul_sign_0 = _mm512_xor_ps(alpha_vec, sign_val_vec_0);
        const __m512 alpha_mul_sign_1 = _mm512_xor_ps(alpha_vec, sign_val_vec_1);

        const __m512 clip_0 = _mm512_mask_blend_ps(abs_lt_alpha_0, val_vec_0, alpha_mul_sign_0);
        const __mmask16 abs_lt_alpha_eps_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_eps_vec_0, _CMP_LT_OQ);
        const __m512 clipped_0 = _mm512_mask_blend_ps(abs_lt_alpha_eps_0, clip_0, _mm512_zeros_ps);

        const __m512 clip_1 = _mm512_mask_blend_ps(abs_lt_alpha_1, val_vec_1, alpha_mul_sign_1);
        const __mmask16 abs_lt_alpha_eps_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_eps_vec_1, _CMP_LT_OQ);
        const __m512 clipped_1 = _mm512_mask_blend_ps(abs_lt_alpha_eps_1, clip_1, _mm512_zeros_ps);

        const __m512 abs_clipped_0 = _mm512_and_ps(clipped_0, _mm512_1st_bit_off);
        const __m512 abs_clipped_1 = _mm512_and_ps(clipped_1, _mm512_1st_bit_off);
        const __m512 abs_clipped_mul_rcp_alpha_0 = _mm512_mul_ps(abs_clipped_0, rcp_alpha_vec);
        const __m512 abs_clipped_mul_rcp_alpha_1 = _mm512_mul_ps(abs_clipped_1, rcp_alpha_vec);

        // TODO: Compute noise
        // TODO: Check for INFINITY!!!!!!!!
        const __m512 log_abs_clipped_rcp_alpha_0 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_0);
        const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
        const __m512 log_abs_clipped_rcp_alpha_1 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_1);
        const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

        // TODO: Expensive truncation, check if there is a better way to do this
        // If log is infinity, the conversion will break, I need to fix that - JUST use _mm512_floor_ps and avoid this whole problem - still a problem for packing
        const __m512 log_trunc_f0 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_0);
        const __m512 log_trunc_f1 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_1);

        const __m512 exp_log_0 = _mm512_qsparse_pow2_ps(log_trunc_f0);
        const __m512 exp_log_1 = _mm512_qsparse_pow2_ps(log_trunc_f1);
        const __m512 noise_0 = _mm512_mul_ps(exp_log_0, rnd_0);
        const __m512 noise_1 = _mm512_mul_ps(exp_log_1, rnd_1);

        // TODO: Might not need to multiply with alpha
        const __m512 dq_threshold_vec_0 = _mm512_mul_ps(exp_log_0, alpha_vec);
        const __m512 dq_threshold_vec_1 = _mm512_mul_ps(exp_log_1, alpha_vec);

        // Compute dequantized val
        const __m512 abs_rnd_tmp_0_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_0, noise_0);
        const __m512 abs_rnd_tmp_0_1 = _mm512_mul_ps(abs_rnd_tmp_0_0, _mm512_4_div_3_ps);
        const __m512 abs_rnd_tmp_1_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_1, noise_1);
        const __m512 abs_rnd_tmp_1_1 = _mm512_mul_ps(abs_rnd_tmp_1_0, _mm512_4_div_3_ps);

        // TODO: Check for INFINITY!!!!!!!!
        const __m512 log_abs_rnd_0 = _mm512_qsparse_log2_ps(abs_rnd_tmp_0_1);
        const __mmask16 neg_inf_rnd_mask_0 = _mm512_cmp_ps_mask(log_abs_rnd_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
        const __m512 log_abs_rnd_1 = _mm512_qsparse_log2_ps(abs_rnd_tmp_1_1);
        const __mmask16 neg_inf_rnd_mask_1 = _mm512_cmp_ps_mask(log_abs_rnd_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

        const __m512 log_trunc_rnd_f0 = _mm512_floor_ps(log_abs_rnd_0);
        const __m512 log_trunc_rnd_f1 = _mm512_floor_ps(log_abs_rnd_1);

        const __m512 exp_log_rnd_0 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f0);
        const __m512 exp_log_rnd_1 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f1);

        const __m512 dq_val_vec_0 = _mm512_mul_ps(exp_log_rnd_0, alpha_vec);
        const __m512 dq_val_vec_1 = _mm512_mul_ps(exp_log_rnd_1, alpha_vec);

        const __mmask16 threshold_mask_0 = _mm512_cmp_ps_mask(dq_val_vec_0, dq_threshold_vec_0, _CMP_LT_OQ);
        const __mmask16 threshold_mask_1 = _mm512_cmp_ps_mask(dq_val_vec_1, dq_threshold_vec_1, _CMP_LT_OQ);
        const __mmask16 zero_mask_0 = _mm512_cmp_ps_mask(val_vec_0, _mm512_zeros_ps, _CMP_EQ_OQ);
        const __mmask16 zero_mask_1 = _mm512_cmp_ps_mask(val_vec_1, _mm512_zeros_ps, _CMP_EQ_OQ);

        // Set -inf to sentinel values
        const __m512i log_trunc_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_0, log_abs_clipped_rcp_alpha_0);
        const __m512i log_trunc_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_1, log_abs_clipped_rcp_alpha_1);
        const __m512i log_trunc_rnd_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_0, log_abs_rnd_0);
        const __m512i log_trunc_rnd_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_1, log_abs_rnd_1);

        const __m512i q_vec_tmp_0 = _mm512_mask_blend_epi32(threshold_mask_0, log_trunc_rnd_i0, log_trunc_i0);
        const __m512i quantized_vec_0 = _mm512_mask_blend_epi32(zero_mask_0, q_vec_tmp_0, _mm512_127_epi32);
        const __m512i q_vec_tmp_1 = _mm512_mask_blend_epi32(threshold_mask_1, log_trunc_rnd_i1, log_trunc_i1);
        const __m512i quantized_vec_1 = _mm512_mask_blend_epi32(zero_mask_1, q_vec_tmp_1, _mm512_127_epi32);

        // Pack and store the exponents
        // TODO: During shifting we have a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3
        // TODO: Try to pack them such that a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d0
        // TODO: During unpacking you can just shift the elements and use a mask to get rid of the extra elements on the left of the target num.
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

        int8_t q_val_1;
        if (val == 0.0f) {
            q_val_1 = 127;
        } else {
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float alpha_rnd = 1.0f;
            const float rnd = 0.0f;
#else
            const float alpha_rnd = get_random_float();
            const float rnd = get_random_float() - 0.5f;
#endif
            // Compute
            const float alpha_eps = alpha * alpha_rnd;
            const float val_abs = fabs(val);
            const float sign = val < 0.0f ? -1.0f : 1.0f;
            const float clip_0 = val_abs < alpha ? alpha * sign : val;
            const float clip_1 = val_abs < alpha_eps ? 0.0f : clip_0;

            // Compute noise
            const float abs_clip = fabs(clip_1);
            const float abs_clip_mul_rcp_alpha = abs_clip * rcp_alpha;
            const float log_abs_clip_mul_rcp_alpha = log2f(abs_clip_mul_rcp_alpha); // TODO: Check for infinity
            const float log_trunc_f = floorf(log_abs_clip_mul_rcp_alpha);
            const float exp_log = exp2f(log_trunc_f);
            const float noise = exp_log * rnd;

            const float dq_threshold_val = exp_log * alpha;

            const float abs_clip_rnd_0 = abs_clip_mul_rcp_alpha + noise;
            const float abs_clip_rnd_1 = abs_clip_rnd_0 * 1.33333333333f;
            const float log_abs_rnd = log2f(abs_clip_rnd_1);
            const float log_rnd_trunc_f = floorf(log_abs_rnd);
            const float exp_log_rnd = exp2f(log_rnd_trunc_f);

            const float dq_val = exp_log_rnd * alpha;

            const int8_t log_trunc_i =
                log_abs_clip_mul_rcp_alpha == NEG_INF ? 127 : (int8_t) log_abs_clip_mul_rcp_alpha;
            const int8_t log_trunc_rnd_i = log_abs_rnd == NEG_INF ? 127 : (int8_t) log_abs_rnd;

            q_val_1 = dq_val < dq_threshold_val ? log_trunc_i : log_trunc_rnd_i;
        }

        // Store
        result[idx] = q_val_1;
    }
}

void quantization::LUQ8Strategy::quantize_parallel(
        union Quantization_Input<int8_t>& input
) {
#if defined(_OPENMP)
    const float* values = input.luq_quantization_input.dq_values;
    int8_t* result = input.luq_quantization_input.q_values;
    const int size = input.luq_quantization_input.size;
    const int mask_count = input.luq_quantization_input.mask_count;
    __mmask16* signs = input.luq_quantization_input.signs;

    const int thread_count = get_OpenMP_threads();
    __m512* acc_max_vecs_0 = new __m512[thread_count];
    __m512* acc_max_vecs_1 = new __m512[thread_count];
    int* signIndices = new int[thread_count];
    // This part is so badly written that I feel the need to explain, mask_incr is equal to number of iterations
    // the parallel loop below will execute multiplied by 2. (The sequential loop would execute size / 32 times)
    // If this is <= to the number of threads then each thread will only execute a single iteration at most, o.w the number
    // of iteration will be divided amongst every thread. Each iteration processes 32 elements, so we will have 2
    // sign masks. Therefore, we multiply the number of iterations per thread with 2 to compute the start index of
    // each sign mask.
    int mask_incr = (size / 32 <= thread_count) ? 2 : (size * 2) / (32 * thread_count);
    for (int thread = 0; thread < thread_count; thread++) {
        acc_max_vecs_0[thread] = _mm512_set1_ps(MIN_FLOAT);
        acc_max_vecs_1[thread] = _mm512_set1_ps(MIN_FLOAT);
        signIndices[thread] = thread * mask_incr;
    }

    // Extract signs and find max(abs(values))
    int idx;
    #pragma omp parallel for default(none) shared(signIndices, size, values, signs, acc_max_vecs_0, acc_max_vecs_1, _mm512_zeros_ps, _mm512_1st_bit_off)
    for (idx = 0; idx < size - 31; idx += 32) {
        const int tid = get_OpenMP_thread();

        // Load
        const __m512 val_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_1 = _mm512_loadu_ps(values + idx + 16);

        // Compute
        const __mmask16 neg_mask_0 = _mm512_cmp_ps_mask(val_0, _mm512_zeros_ps, _CMP_LT_OQ);
        const __mmask16 neg_mask_1 = _mm512_cmp_ps_mask(val_1, _mm512_zeros_ps, _CMP_LT_OQ);
        const __m512 abs_val_0 = _mm512_and_ps(val_0, _mm512_1st_bit_off);
        const __m512 abs_val_1 = _mm512_and_ps(val_1, _mm512_1st_bit_off);
        const __m512 max_vec_0 = _mm512_max_ps(abs_val_0, acc_max_vecs_0[tid]);
        const __m512 max_vec_1 = _mm512_max_ps(abs_val_1, acc_max_vecs_1[tid]);

        // Store
        acc_max_vecs_0[tid] = max_vec_0;
        acc_max_vecs_1[tid] = max_vec_1;
        signs[signIndices[tid]] = neg_mask_0;
        signs[signIndices[tid] + 1] = neg_mask_1;
        signIndices[tid] += 2;
    }
    __m512 acc_max_0 = _mm512_set1_ps(MIN_FLOAT);
    __m512 acc_max_1 = _mm512_set1_ps(MIN_FLOAT);
    for (int thread = 0; thread < thread_count; thread++) {
        acc_max_0 = _mm512_max_ps(acc_max_0, acc_max_vecs_0[thread]);
        acc_max_1 = _mm512_max_ps(acc_max_1, acc_max_vecs_1[thread]);
    }
    delete [] acc_max_vecs_0;
    delete [] acc_max_vecs_1;
    delete [] signIndices;
    const float max_0 = _mm512_hmaxf32_ss(acc_max_0);
    const float max_1 = _mm512_hmaxf32_ss(acc_max_1);
    float max = max_0 <= max_1 ? max_1 : max_0;

    uint32_t mask = 0;
    idx = (size / 32) * 32;
    for (int bitIdx = 0; idx < size; idx++, bitIdx++) {
        // Load
        const float val = fabs(values[idx]);

        // Compute
        const float curr_max = max <= val ? val : max;
        const uint32_t bit = values[idx] < 0 ? (1 << bitIdx) : 0; // TODO: Try using (values[idx] < 0) << bitIdx

        // Store
        max = curr_max;
        mask = mask | bit;
    }
    int signIdx = (size / 32) * 2;
    if (signIdx < mask_count) {
        signs[signIdx++] = mask;
    }
    if (signIdx < mask_count) {
        signs[signIdx] = mask >> 16;
    }
    const float alpha = max * threshold_denom;
    const float rcp_alpha = 1.0f / alpha;
    input.luq_quantization_input.scale = alpha;

    const __m512 alpha_vec = _mm512_set1_ps(alpha);
    const __m512 rcp_alpha_vec = _mm512_set1_ps(1.0f / alpha);
    #pragma omp parallel for default(none) shared(size, values, alpha_vec, _mm512_1st_bit_off, _mm512_1st_bit_on, _mm512_zeros_ps, rcp_alpha_vec, _mm512_neg_inf_ps, _mm512_4_div_3_ps, _mm512_127_epi32, result, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, _mm512_0_5_ps, avx512_random_key1_perthread, avx512_random_key2_perthread)
    for (idx = 0; idx < size - 31; idx += 32) {
        const __m512 val_vec_0 = _mm512_loadu_ps(values + idx);
        const __m512 val_vec_1 = _mm512_loadu_ps(values + idx + 16);

#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
        const __m512 alpha_rnd_0 = _mm512_set1_ps(1.0f);
        const __m512 alpha_rnd_1 = _mm512_set1_ps(1.0f);
        const __m512 rnd_0 = _mm512_setzero_ps();
        const __m512 rnd_1 = _mm512_setzero_ps();
#else
        const int tid = get_OpenMP_thread();

        const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1_perthread[tid], avx512_random_key2_perthread[tid]);

        const __m512i alpha_rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
        const __m512i alpha_rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 8);
        const __m512i rnd_i8_0 = _mm512_slli_epi32(alpha_rnd_i8_0, 16);
        const __m512i rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 24);

        const __m512 alpha_rnd_f8_0 = _mm512_cvtepi32_ps(alpha_rnd_i8_0);
        const __m512 alpha_rnd_f8_1 = _mm512_cvtepi32_ps(alpha_rnd_i8_1);
        const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
        const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

        const __m512 alpha_rnd_0 = _mm512_mul_ps(alpha_rnd_f8_0, _mm512_rcp_2pow31_ps);
        const __m512 alpha_rnd_1 = _mm512_mul_ps(alpha_rnd_f8_1, _mm512_rcp_2pow31_ps);
        const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
        const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
        const __m512 alpha_eps_vec_0 = _mm512_mul_ps(alpha_vec, alpha_rnd_0);
        const __m512 alpha_eps_vec_1 = _mm512_mul_ps(alpha_vec, alpha_rnd_1);

        const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
        const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

        const __m512 sign_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_on);
        const __m512 sign_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_on);

        const __mmask16 abs_lt_alpha_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_vec, _CMP_LT_OQ);
        const __mmask16 abs_lt_alpha_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_vec, _CMP_LT_OQ);
        const __m512 alpha_mul_sign_0 = _mm512_xor_ps(alpha_vec, sign_val_vec_0);
        const __m512 alpha_mul_sign_1 = _mm512_xor_ps(alpha_vec, sign_val_vec_1);

        const __m512 clip_0 = _mm512_mask_blend_ps(abs_lt_alpha_0, val_vec_0, alpha_mul_sign_0);
        const __mmask16 abs_lt_alpha_eps_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_eps_vec_0, _CMP_LT_OQ);
        const __m512 clipped_0 = _mm512_mask_blend_ps(abs_lt_alpha_eps_0, clip_0, _mm512_zeros_ps);

        const __m512 clip_1 = _mm512_mask_blend_ps(abs_lt_alpha_1, val_vec_1, alpha_mul_sign_1);
        const __mmask16 abs_lt_alpha_eps_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_eps_vec_1, _CMP_LT_OQ);
        const __m512 clipped_1 = _mm512_mask_blend_ps(abs_lt_alpha_eps_1, clip_1, _mm512_zeros_ps);

        const __m512 abs_clipped_0 = _mm512_and_ps(clipped_0, _mm512_1st_bit_off);
        const __m512 abs_clipped_1 = _mm512_and_ps(clipped_1, _mm512_1st_bit_off);
        const __m512 abs_clipped_mul_rcp_alpha_0 = _mm512_mul_ps(abs_clipped_0, rcp_alpha_vec);
        const __m512 abs_clipped_mul_rcp_alpha_1 = _mm512_mul_ps(abs_clipped_1, rcp_alpha_vec);

        // TODO: Compute noise
        // TODO: Check for INFINITY!!!!!!!!
        const __m512 log_abs_clipped_rcp_alpha_0 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_0);
        const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
        const __m512 log_abs_clipped_rcp_alpha_1 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_1);
        const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

        // TODO: Expensive truncation, check if there is a better way to do this
        // If log is infinity, the conversion will break, I need to fix that - JUST use _mm512_floor_ps and avoid this whole problem - still a problem for packing
        const __m512 log_trunc_f0 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_0);
        const __m512 log_trunc_f1 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_1);

        const __m512 exp_log_0 = _mm512_qsparse_pow2_ps(log_trunc_f0);
        const __m512 exp_log_1 = _mm512_qsparse_pow2_ps(log_trunc_f1);
        const __m512 noise_0 = _mm512_mul_ps(exp_log_0, rnd_0);
        const __m512 noise_1 = _mm512_mul_ps(exp_log_1, rnd_1);

        // TODO: Might not need to multiply with alpha
        const __m512 dq_threshold_vec_0 = _mm512_mul_ps(exp_log_0, alpha_vec);
        const __m512 dq_threshold_vec_1 = _mm512_mul_ps(exp_log_1, alpha_vec);

        // Compute dequantized val
        const __m512 abs_rnd_tmp_0_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_0, noise_0);
        const __m512 abs_rnd_tmp_0_1 = _mm512_mul_ps(abs_rnd_tmp_0_0, _mm512_4_div_3_ps);
        const __m512 abs_rnd_tmp_1_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_1, noise_1);
        const __m512 abs_rnd_tmp_1_1 = _mm512_mul_ps(abs_rnd_tmp_1_0, _mm512_4_div_3_ps);

        // TODO: Check for INFINITY!!!!!!!!
        const __m512 log_abs_rnd_0 = _mm512_qsparse_log2_ps(abs_rnd_tmp_0_1);
        const __mmask16 neg_inf_rnd_mask_0 = _mm512_cmp_ps_mask(log_abs_rnd_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
        const __m512 log_abs_rnd_1 = _mm512_qsparse_log2_ps(abs_rnd_tmp_1_1);
        const __mmask16 neg_inf_rnd_mask_1 = _mm512_cmp_ps_mask(log_abs_rnd_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

        const __m512 log_trunc_rnd_f0 = _mm512_floor_ps(log_abs_rnd_0);
        const __m512 log_trunc_rnd_f1 = _mm512_floor_ps(log_abs_rnd_1);

        const __m512 exp_log_rnd_0 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f0);
        const __m512 exp_log_rnd_1 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f1);

        const __m512 dq_val_vec_0 = _mm512_mul_ps(exp_log_rnd_0, alpha_vec);
        const __m512 dq_val_vec_1 = _mm512_mul_ps(exp_log_rnd_1, alpha_vec);

        const __mmask16 threshold_mask_0 = _mm512_cmp_ps_mask(dq_val_vec_0, dq_threshold_vec_0, _CMP_LT_OQ);
        const __mmask16 threshold_mask_1 = _mm512_cmp_ps_mask(dq_val_vec_1, dq_threshold_vec_1, _CMP_LT_OQ);
        const __mmask16 zero_mask_0 = _mm512_cmp_ps_mask(val_vec_0, _mm512_zeros_ps, _CMP_EQ_OQ);
        const __mmask16 zero_mask_1 = _mm512_cmp_ps_mask(val_vec_1, _mm512_zeros_ps, _CMP_EQ_OQ);

        // Set -inf to sentinel values
        const __m512i log_trunc_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_0, log_abs_clipped_rcp_alpha_0);
        const __m512i log_trunc_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_1, log_abs_clipped_rcp_alpha_1);
        const __m512i log_trunc_rnd_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_0, log_abs_rnd_0);
        const __m512i log_trunc_rnd_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_1, log_abs_rnd_1);

        const __m512i q_vec_tmp_0 = _mm512_mask_blend_epi32(threshold_mask_0, log_trunc_rnd_i0, log_trunc_i0);
        const __m512i quantized_vec_0 = _mm512_mask_blend_epi32(zero_mask_0, q_vec_tmp_0, _mm512_127_epi32);
        const __m512i q_vec_tmp_1 = _mm512_mask_blend_epi32(threshold_mask_1, log_trunc_rnd_i1, log_trunc_i1);
        const __m512i quantized_vec_1 = _mm512_mask_blend_epi32(zero_mask_1, q_vec_tmp_1, _mm512_127_epi32);

        // Pack and store the exponents
        // TODO: During shifting we have a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3
        // TODO: Try to pack them such that a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d0
        // TODO: During unpacking you can just shift the elements and use a mask to get rid of the extra elements on the left of the target num.
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
    #pragma omp parallel for default(none) shared(idx, size, values, alpha, rcp_alpha, NEG_INF, result)
    for (int idx2 = idx; idx2 < size; idx2++) {
        // Load
        const float val = values[idx2];

        int8_t q_val_1;
        if (val == 0.0f) {
            q_val_1 = 127;
        } else {
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const float alpha_rnd = 1.0f;
            const float rnd = 0.0f;
#else
            const float alpha_rnd = get_random_float();
            const float rnd = get_random_float() - 0.5f;
#endif
            // Compute
            const float alpha_eps = alpha * alpha_rnd;
            const float val_abs = fabs(val);
            const float sign = val < 0.0f ? -1.0f : 1.0f;
            const float clip_0 = val_abs < alpha ? alpha * sign : val;
            const float clip_1 = val_abs < alpha_eps ? 0.0f : clip_0;

            // Compute noise
            const float abs_clip = fabs(clip_1);
            const float abs_clip_mul_rcp_alpha = abs_clip * rcp_alpha;
            const float log_abs_clip_mul_rcp_alpha = log2f(abs_clip_mul_rcp_alpha); // TODO: Check for infinity
            const float log_trunc_f = floorf(log_abs_clip_mul_rcp_alpha);
            const float exp_log = exp2f(log_trunc_f);
            const float noise = exp_log * rnd;

            const float dq_threshold_val = exp_log * alpha;

            const float abs_clip_rnd_0 = abs_clip_mul_rcp_alpha + noise;
            const float abs_clip_rnd_1 = abs_clip_rnd_0 * 1.33333333333f;
            const float log_abs_rnd = log2f(abs_clip_rnd_1);
            const float log_rnd_trunc_f = floorf(log_abs_rnd);
            const float exp_log_rnd = exp2f(log_rnd_trunc_f);

            const float dq_val = exp_log_rnd * alpha;

            const int8_t log_trunc_i =
                log_abs_clip_mul_rcp_alpha == NEG_INF ? 127 : (int8_t) log_abs_clip_mul_rcp_alpha;
            const int8_t log_trunc_rnd_i = log_abs_rnd == NEG_INF ? 127 : (int8_t) log_abs_rnd;

            q_val_1 = dq_val < dq_threshold_val ? log_trunc_i : log_trunc_rnd_i;
        }

        // Store
        result[idx2] = q_val_1;
    }
#else
    quantization::LUQ8Strategy::quantize(input);
#endif
}

// IMPORTANT: Group size has to be at least 16, o.w. the signs work incorrectly!!!!!!
void quantization::LUQ8Strategy::quantize_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.luq_grouped_input.dq_values;
    int8_t* result = input.luq_grouped_input.q_values;
    const int size = input.luq_grouped_input.size;
    float* scale = input.luq_grouped_input.scale;
    const int mask_count = input.luq_grouped_input.mask_count;
    __mmask16* signs = input.luq_grouped_input.signs;

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
        const int group_mask_count = (group == group_count - 1)
             ? (group_size % qgroup_size) == 0
               ? (group_size >> 4)
               : (group_size + 15) / 16
             : (group_size >> 4);
        const int mask_start = group * (qgroup_size >> 4);
        __mmask16* group_signs = signs + mask_start;

        __m512 acc_max_0 = _mm512_set1_ps(MIN_FLOAT);
        __m512 acc_max_1 = _mm512_set1_ps(MIN_FLOAT);
        // Extract signs and find max(abs(values))
        int idx;
        int signIdx;
        for (idx = 0, signIdx = 0; idx < group_size - 31; idx += 32, signIdx += 2) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_0 = _mm512_loadu_ps(group_base);
            const __m512 val_1 = _mm512_loadu_ps(group_base + 16);

            // Compute
            const __mmask16 neg_mask_0 = _mm512_cmp_ps_mask(val_0, _mm512_zeros_ps, _CMP_LT_OQ);
            const __mmask16 neg_mask_1 = _mm512_cmp_ps_mask(val_1, _mm512_zeros_ps, _CMP_LT_OQ);
            const __m512 abs_val_0 = _mm512_and_ps(val_0, _mm512_1st_bit_off);
            const __m512 abs_val_1 = _mm512_and_ps(val_1, _mm512_1st_bit_off);
            const __m512 max_vec_0 = _mm512_max_ps(abs_val_0, acc_max_0);
            const __m512 max_vec_1 = _mm512_max_ps(abs_val_1, acc_max_1);

            // Store
            acc_max_0 = max_vec_0;
            acc_max_1 = max_vec_1;
            group_signs[signIdx] = neg_mask_0;
            group_signs[signIdx + 1] = neg_mask_1;
        }
        const float max_0 = _mm512_hmaxf32_ss(acc_max_0);
        const float max_1 = _mm512_hmaxf32_ss(acc_max_1);
        float max = max_0 <= max_1 ? max_1 : max_0;

        uint32_t mask = 0;
        int bitIdx;
        for (bitIdx = 0; idx < group_size; idx++, bitIdx++) {
            // Load
            const float val = fabs(group_values[idx]);

            // Compute
            const float curr_max = max <= val ? val : max;
            const uint32_t bit = group_values[idx] < 0 ? (1 << bitIdx) : 0; // TODO: Try using (values[idx] < 0) << bitIdx

            // Store
            max = curr_max;
            mask = mask | bit;
        }
        if (signIdx < group_mask_count) {
            group_signs[signIdx++] = mask;
        }
        if (signIdx < group_mask_count) {
            group_signs[signIdx] = mask >> 16;
        }
        const float alpha = max * threshold_denom;
        const float rcp_alpha = 1.0f / alpha;
        scale[group] = alpha;

        const __m512 alpha_vec = _mm512_set1_ps(alpha);
        const __m512 rcp_alpha_vec = _mm512_set1_ps(rcp_alpha);
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const __m512 alpha_rnd_0 = _mm512_set1_ps(1.0f);
            const __m512 alpha_rnd_1 = _mm512_set1_ps(1.0f);
            const __m512 rnd_0 = _mm512_setzero_ps();
            const __m512 rnd_1 = _mm512_setzero_ps();
#else
            const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1, avx512_random_key2);

            const __m512i alpha_rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
            const __m512i alpha_rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 8);
            const __m512i rnd_i8_0 = _mm512_slli_epi32(alpha_rnd_i8_0, 16);
            const __m512i rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 24);

            const __m512 alpha_rnd_f8_0 = _mm512_cvtepi32_ps(alpha_rnd_i8_0);
            const __m512 alpha_rnd_f8_1 = _mm512_cvtepi32_ps(alpha_rnd_i8_1);
            const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
            const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

            const __m512 alpha_rnd_0 = _mm512_mul_ps(alpha_rnd_f8_0, _mm512_rcp_2pow31_ps);
            const __m512 alpha_rnd_1 = _mm512_mul_ps(alpha_rnd_f8_1, _mm512_rcp_2pow31_ps);
            const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
            const __m512 alpha_eps_vec_0 = _mm512_mul_ps(alpha_vec, alpha_rnd_0);
            const __m512 alpha_eps_vec_1 = _mm512_mul_ps(alpha_vec, alpha_rnd_1);

            const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
            const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

            const __m512 sign_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_on);
            const __m512 sign_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_on);

            const __mmask16 abs_lt_alpha_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_vec, _CMP_LT_OQ);
            const __mmask16 abs_lt_alpha_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_vec, _CMP_LT_OQ);
            const __m512 alpha_mul_sign_0 = _mm512_xor_ps(alpha_vec, sign_val_vec_0);
            const __m512 alpha_mul_sign_1 = _mm512_xor_ps(alpha_vec, sign_val_vec_1);

            const __m512 clip_0 = _mm512_mask_blend_ps(abs_lt_alpha_0, val_vec_0, alpha_mul_sign_0);
            const __mmask16 abs_lt_alpha_eps_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_eps_vec_0, _CMP_LT_OQ);
            const __m512 clipped_0 = _mm512_mask_blend_ps(abs_lt_alpha_eps_0, clip_0, _mm512_zeros_ps);

            const __m512 clip_1 = _mm512_mask_blend_ps(abs_lt_alpha_1, val_vec_1, alpha_mul_sign_1);
            const __mmask16 abs_lt_alpha_eps_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_eps_vec_1, _CMP_LT_OQ);
            const __m512 clipped_1 = _mm512_mask_blend_ps(abs_lt_alpha_eps_1, clip_1, _mm512_zeros_ps);

            const __m512 abs_clipped_0 = _mm512_and_ps(clipped_0, _mm512_1st_bit_off);
            const __m512 abs_clipped_1 = _mm512_and_ps(clipped_1, _mm512_1st_bit_off);
            const __m512 abs_clipped_mul_rcp_alpha_0 = _mm512_mul_ps(abs_clipped_0, rcp_alpha_vec);
            const __m512 abs_clipped_mul_rcp_alpha_1 = _mm512_mul_ps(abs_clipped_1, rcp_alpha_vec);

            // TODO: Compute noise
            // TODO: Check for INFINITY!!!!!!!!
            const __m512 log_abs_clipped_rcp_alpha_0 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_0);
            const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
            const __m512 log_abs_clipped_rcp_alpha_1 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_1);
            const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

            // TODO: Expensive truncation, check if there is a better way to do this
            // If log is infinity, the conversion will break, I need to fix that - JUST use _mm512_floor_ps and avoid this whole problem - still a problem for packing
            const __m512 log_trunc_f0 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_0);
            const __m512 log_trunc_f1 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_1);

            const __m512 exp_log_0 = _mm512_qsparse_pow2_ps(log_trunc_f0);
            const __m512 exp_log_1 = _mm512_qsparse_pow2_ps(log_trunc_f1);
            const __m512 noise_0 = _mm512_mul_ps(exp_log_0, rnd_0);
            const __m512 noise_1 = _mm512_mul_ps(exp_log_1, rnd_1);

            // TODO: Might not need to multiply with alpha
            const __m512 dq_threshold_vec_0 = _mm512_mul_ps(exp_log_0, alpha_vec);
            const __m512 dq_threshold_vec_1 = _mm512_mul_ps(exp_log_1, alpha_vec);

            // Compute dequantized val
            const __m512 abs_rnd_tmp_0_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_0, noise_0);
            const __m512 abs_rnd_tmp_0_1 = _mm512_mul_ps(abs_rnd_tmp_0_0, _mm512_4_div_3_ps);
            const __m512 abs_rnd_tmp_1_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_1, noise_1);
            const __m512 abs_rnd_tmp_1_1 = _mm512_mul_ps(abs_rnd_tmp_1_0, _mm512_4_div_3_ps);

            // TODO: Check for INFINITY!!!!!!!!
            const __m512 log_abs_rnd_0 = _mm512_qsparse_log2_ps(abs_rnd_tmp_0_1);
            const __mmask16 neg_inf_rnd_mask_0 = _mm512_cmp_ps_mask(log_abs_rnd_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
            const __m512 log_abs_rnd_1 = _mm512_qsparse_log2_ps(abs_rnd_tmp_1_1);
            const __mmask16 neg_inf_rnd_mask_1 = _mm512_cmp_ps_mask(log_abs_rnd_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

            const __m512 log_trunc_rnd_f0 = _mm512_floor_ps(log_abs_rnd_0);
            const __m512 log_trunc_rnd_f1 = _mm512_floor_ps(log_abs_rnd_1);

            const __m512 exp_log_rnd_0 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f0);
            const __m512 exp_log_rnd_1 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f1);

            const __m512 dq_val_vec_0 = _mm512_mul_ps(exp_log_rnd_0, alpha_vec);
            const __m512 dq_val_vec_1 = _mm512_mul_ps(exp_log_rnd_1, alpha_vec);

            const __mmask16 threshold_mask_0 = _mm512_cmp_ps_mask(dq_val_vec_0, dq_threshold_vec_0, _CMP_LT_OQ);
            const __mmask16 threshold_mask_1 = _mm512_cmp_ps_mask(dq_val_vec_1, dq_threshold_vec_1, _CMP_LT_OQ);
            const __mmask16 zero_mask_0 = _mm512_cmp_ps_mask(val_vec_0, _mm512_zeros_ps, _CMP_EQ_OQ);
            const __mmask16 zero_mask_1 = _mm512_cmp_ps_mask(val_vec_1, _mm512_zeros_ps, _CMP_EQ_OQ);

            // Set -inf to sentinel values
            const __m512i log_trunc_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_0, log_abs_clipped_rcp_alpha_0);
            const __m512i log_trunc_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_1, log_abs_clipped_rcp_alpha_1);
            const __m512i log_trunc_rnd_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_0, log_abs_rnd_0);
            const __m512i log_trunc_rnd_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_1, log_abs_rnd_1);

            const __m512i q_vec_tmp_0 = _mm512_mask_blend_epi32(threshold_mask_0, log_trunc_rnd_i0, log_trunc_i0);
            const __m512i quantized_vec_0 = _mm512_mask_blend_epi32(zero_mask_0, q_vec_tmp_0, _mm512_127_epi32);
            const __m512i q_vec_tmp_1 = _mm512_mask_blend_epi32(threshold_mask_1, log_trunc_rnd_i1, log_trunc_i1);
            const __m512i quantized_vec_1 = _mm512_mask_blend_epi32(zero_mask_1, q_vec_tmp_1, _mm512_127_epi32);

            // Pack and store the exponents
            // TODO: During shifting we have a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3
            // TODO: Try to pack them such that a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d0
            // TODO: During unpacking you can just shift the elements and use a mask to get rid of the extra elements on the left of the target num.
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

            int8_t q_val_1;
            if (val == 0.0f) {
                q_val_1 = 127;
            } else {
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
                const float alpha_rnd = 1.0f;
                const float rnd = 0.0f;
#else
                const float alpha_rnd = get_random_float();
                const float rnd = get_random_float() - 0.5f;
#endif
                // Compute
                const float alpha_eps = alpha * alpha_rnd;
                const float val_abs = fabs(val);
                const float sign = val < 0.0f ? -1.0f : 1.0f;
                const float clip_0 = val_abs < alpha ? alpha * sign : val;
                const float clip_1 = val_abs < alpha_eps ? 0.0f : clip_0;

                // Compute noise
                const float abs_clip = fabs(clip_1);
                const float abs_clip_mul_rcp_alpha = abs_clip * rcp_alpha;
                const float log_abs_clip_mul_rcp_alpha = log2f(abs_clip_mul_rcp_alpha); // TODO: Check for infinity
                const float log_trunc_f = floorf(log_abs_clip_mul_rcp_alpha);
                const float exp_log = exp2f(log_trunc_f);
                const float noise = exp_log * rnd;

                const float dq_threshold_val = exp_log * alpha;

                const float abs_clip_rnd_0 = abs_clip_mul_rcp_alpha + noise;
                const float abs_clip_rnd_1 = abs_clip_rnd_0 * 1.33333333333f;
                const float log_abs_rnd = log2f(abs_clip_rnd_1);
                const float log_rnd_trunc_f = floorf(log_abs_rnd);
                const float exp_log_rnd = exp2f(log_rnd_trunc_f);

                const float dq_val = exp_log_rnd * alpha;

                const int8_t log_trunc_i =
                        log_abs_clip_mul_rcp_alpha == NEG_INF ? 127 : (int8_t) log_abs_clip_mul_rcp_alpha;
                const int8_t log_trunc_rnd_i = log_abs_rnd == NEG_INF ? 127 : (int8_t) log_abs_rnd;

                q_val_1 = dq_val < dq_threshold_val ? log_trunc_i : log_trunc_rnd_i;
            }

            // Store
            group_result[idx] = q_val_1;
        }
    }
}

void quantization::LUQ8Strategy::quantize_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const float* values = input.luq_grouped_input.dq_values;
    int8_t* result = input.luq_grouped_input.q_values;
    const int size = input.luq_grouped_input.size;
    float* scale = input.luq_grouped_input.scale;
    const int mask_count = input.luq_grouped_input.mask_count;
    __mmask16* signs = input.luq_grouped_input.signs;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, signs, MIN_FLOAT, _mm512_zeros_ps, _mm512_1st_bit_off, _mm512_1st_bit_off_epi8, _mm512_rcp_2pow31_ps, scale, _mm512_0_5_ps, _mm512_1st_bit_on, _mm512_neg_inf_ps, _mm512_4_div_3_ps, _mm512_127_epi32, NEG_INF, avx512_random_key1_perthread, avx512_random_key2_perthread)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
            ? (size % qgroup_size) == 0
                ? qgroup_size
                : size % qgroup_size
            : qgroup_size;
        const int group_start = group * qgroup_size;
        const float* group_values = values + group_start;
        int8_t* group_result = result + group_start;
        const int group_mask_count = (group == group_count - 1)
            ? (group_size % qgroup_size) == 0
                ? (group_size >> 4)
                : (group_size + 15) / 16
            : (group_size >> 4);
        const int mask_start = group * (qgroup_size >> 4);
        __mmask16* group_signs = signs + mask_start;

        __m512 acc_max_0 = _mm512_set1_ps(MIN_FLOAT);
        __m512 acc_max_1 = _mm512_set1_ps(MIN_FLOAT);
        // Extract signs and find max(abs(values))
        int idx;
        int signIdx;
        for (idx = 0, signIdx = 0; idx < group_size - 31; idx += 32, signIdx += 2) {
            // Load
            const float* group_base = group_values + idx;
            const __m512 val_0 = _mm512_loadu_ps(group_base);
            const __m512 val_1 = _mm512_loadu_ps(group_base + 16);

            // Compute
            const __mmask16 neg_mask_0 = _mm512_cmp_ps_mask(val_0, _mm512_zeros_ps, _CMP_LT_OQ);
            const __mmask16 neg_mask_1 = _mm512_cmp_ps_mask(val_1, _mm512_zeros_ps, _CMP_LT_OQ);
            const __m512 abs_val_0 = _mm512_and_ps(val_0, _mm512_1st_bit_off);
            const __m512 abs_val_1 = _mm512_and_ps(val_1, _mm512_1st_bit_off);
            const __m512 max_vec_0 = _mm512_max_ps(abs_val_0, acc_max_0);
            const __m512 max_vec_1 = _mm512_max_ps(abs_val_1, acc_max_1);

            // Store
            acc_max_0 = max_vec_0;
            acc_max_1 = max_vec_1;
            group_signs[signIdx] = neg_mask_0;
            group_signs[signIdx + 1] = neg_mask_1;
        }
        const float max_0 = _mm512_hmaxf32_ss(acc_max_0);
        const float max_1 = _mm512_hmaxf32_ss(acc_max_1);
        float max = max_0 <= max_1 ? max_1 : max_0;

        uint32_t mask = 0;
        int bitIdx;
        for (bitIdx = 0; idx < group_size; idx++, bitIdx++) {
            // Load
            const float val = fabs(group_values[idx]);

            // Compute
            const float curr_max = max <= val ? val : max;
            const uint32_t bit = group_values[idx] < 0 ? (1 << bitIdx) : 0; // TODO: Try using (values[idx] < 0) << bitIdx

            // Store
            max = curr_max;
            mask = mask | bit;
        }
        if (signIdx < group_mask_count) {
            group_signs[signIdx++] = mask;
        }
        if (signIdx < group_mask_count) {
            group_signs[signIdx] = mask >> 16;
        }
        const float alpha = max * threshold_denom;
        const float rcp_alpha = 1.0f / alpha;
        scale[group] = alpha;

        const __m512 alpha_vec = _mm512_set1_ps(alpha);
        const __m512 rcp_alpha_vec = _mm512_set1_ps(rcp_alpha);
        for (idx = 0; idx < group_size - 31; idx += 32) {
            const float* group_base = group_values + idx;
            const __m512 val_vec_0 = _mm512_loadu_ps(group_base);
            const __m512 val_vec_1 = _mm512_loadu_ps(group_base + 16);

#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
            const __m512 alpha_rnd_0 = _mm512_set1_ps(1.0f);
            const __m512 alpha_rnd_1 = _mm512_set1_ps(1.0f);
            const __m512 rnd_0 = _mm512_setzero_ps();
            const __m512 rnd_1 = _mm512_setzero_ps();
#else
            const int tid = get_OpenMP_thread();

            const __m512i rnd_xor_0 = avx512_xorshift128plus(avx512_random_key1_perthread[tid], avx512_random_key2_perthread[tid]);

            const __m512i alpha_rnd_i8_0 = _mm512_and_si512(rnd_xor_0, _mm512_1st_bit_off_epi8);
            const __m512i alpha_rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 8);
            const __m512i rnd_i8_0 = _mm512_slli_epi32(alpha_rnd_i8_0, 16);
            const __m512i rnd_i8_1 = _mm512_slli_epi32(alpha_rnd_i8_0, 24);

            const __m512 alpha_rnd_f8_0 = _mm512_cvtepi32_ps(alpha_rnd_i8_0);
            const __m512 alpha_rnd_f8_1 = _mm512_cvtepi32_ps(alpha_rnd_i8_1);
            const __m512 rnd_f8_0 = _mm512_cvtepi32_ps(rnd_i8_0);
            const __m512 rnd_f8_1 = _mm512_cvtepi32_ps(rnd_i8_1);

            const __m512 alpha_rnd_0 = _mm512_mul_ps(alpha_rnd_f8_0, _mm512_rcp_2pow31_ps);
            const __m512 alpha_rnd_1 = _mm512_mul_ps(alpha_rnd_f8_1, _mm512_rcp_2pow31_ps);
            const __m512 rnd_0 = _mm512_fmsub_ps(rnd_f8_0, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
            const __m512 rnd_1 = _mm512_fmsub_ps(rnd_f8_1, _mm512_rcp_2pow31_ps, _mm512_0_5_ps);
#endif
            const __m512 alpha_eps_vec_0 = _mm512_mul_ps(alpha_vec, alpha_rnd_0);
            const __m512 alpha_eps_vec_1 = _mm512_mul_ps(alpha_vec, alpha_rnd_1);

            const __m512 abs_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_off);
            const __m512 abs_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_off);

            const __m512 sign_val_vec_0 = _mm512_and_ps(val_vec_0, _mm512_1st_bit_on);
            const __m512 sign_val_vec_1 = _mm512_and_ps(val_vec_1, _mm512_1st_bit_on);

            const __mmask16 abs_lt_alpha_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_vec, _CMP_LT_OQ);
            const __mmask16 abs_lt_alpha_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_vec, _CMP_LT_OQ);
            const __m512 alpha_mul_sign_0 = _mm512_xor_ps(alpha_vec, sign_val_vec_0);
            const __m512 alpha_mul_sign_1 = _mm512_xor_ps(alpha_vec, sign_val_vec_1);

            const __m512 clip_0 = _mm512_mask_blend_ps(abs_lt_alpha_0, val_vec_0, alpha_mul_sign_0);
            const __mmask16 abs_lt_alpha_eps_0 = _mm512_cmp_ps_mask(abs_val_vec_0, alpha_eps_vec_0, _CMP_LT_OQ);
            const __m512 clipped_0 = _mm512_mask_blend_ps(abs_lt_alpha_eps_0, clip_0, _mm512_zeros_ps);

            const __m512 clip_1 = _mm512_mask_blend_ps(abs_lt_alpha_1, val_vec_1, alpha_mul_sign_1);
            const __mmask16 abs_lt_alpha_eps_1 = _mm512_cmp_ps_mask(abs_val_vec_1, alpha_eps_vec_1, _CMP_LT_OQ);
            const __m512 clipped_1 = _mm512_mask_blend_ps(abs_lt_alpha_eps_1, clip_1, _mm512_zeros_ps);

            const __m512 abs_clipped_0 = _mm512_and_ps(clipped_0, _mm512_1st_bit_off);
            const __m512 abs_clipped_1 = _mm512_and_ps(clipped_1, _mm512_1st_bit_off);
            const __m512 abs_clipped_mul_rcp_alpha_0 = _mm512_mul_ps(abs_clipped_0, rcp_alpha_vec);
            const __m512 abs_clipped_mul_rcp_alpha_1 = _mm512_mul_ps(abs_clipped_1, rcp_alpha_vec);

            // TODO: Compute noise
            // TODO: Check for INFINITY!!!!!!!!
            const __m512 log_abs_clipped_rcp_alpha_0 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_0);
            const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
            const __m512 log_abs_clipped_rcp_alpha_1 = _mm512_qsparse_log2_ps(abs_clipped_mul_rcp_alpha_1);
            const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(log_abs_clipped_rcp_alpha_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

            // TODO: Expensive truncation, check if there is a better way to do this
            // If log is infinity, the conversion will break, I need to fix that - JUST use _mm512_floor_ps and avoid this whole problem - still a problem for packing
            const __m512 log_trunc_f0 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_0);
            const __m512 log_trunc_f1 = _mm512_floor_ps(log_abs_clipped_rcp_alpha_1);

            const __m512 exp_log_0 = _mm512_qsparse_pow2_ps(log_trunc_f0);
            const __m512 exp_log_1 = _mm512_qsparse_pow2_ps(log_trunc_f1);
            const __m512 noise_0 = _mm512_mul_ps(exp_log_0, rnd_0);
            const __m512 noise_1 = _mm512_mul_ps(exp_log_1, rnd_1);

            // TODO: Might not need to multiply with alpha
            const __m512 dq_threshold_vec_0 = _mm512_mul_ps(exp_log_0, alpha_vec);
            const __m512 dq_threshold_vec_1 = _mm512_mul_ps(exp_log_1, alpha_vec);

            // Compute dequantized val
            const __m512 abs_rnd_tmp_0_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_0, noise_0);
            const __m512 abs_rnd_tmp_0_1 = _mm512_mul_ps(abs_rnd_tmp_0_0, _mm512_4_div_3_ps);
            const __m512 abs_rnd_tmp_1_0 = _mm512_add_ps(abs_clipped_mul_rcp_alpha_1, noise_1);
            const __m512 abs_rnd_tmp_1_1 = _mm512_mul_ps(abs_rnd_tmp_1_0, _mm512_4_div_3_ps);

            // TODO: Check for INFINITY!!!!!!!!
            const __m512 log_abs_rnd_0 = _mm512_qsparse_log2_ps(abs_rnd_tmp_0_1);
            const __mmask16 neg_inf_rnd_mask_0 = _mm512_cmp_ps_mask(log_abs_rnd_0, _mm512_neg_inf_ps, _CMP_NEQ_OQ);
            const __m512 log_abs_rnd_1 = _mm512_qsparse_log2_ps(abs_rnd_tmp_1_1);
            const __mmask16 neg_inf_rnd_mask_1 = _mm512_cmp_ps_mask(log_abs_rnd_1, _mm512_neg_inf_ps, _CMP_NEQ_OQ);

            const __m512 log_trunc_rnd_f0 = _mm512_floor_ps(log_abs_rnd_0);
            const __m512 log_trunc_rnd_f1 = _mm512_floor_ps(log_abs_rnd_1);

            const __m512 exp_log_rnd_0 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f0);
            const __m512 exp_log_rnd_1 = _mm512_qsparse_pow2_ps(log_trunc_rnd_f1);

            const __m512 dq_val_vec_0 = _mm512_mul_ps(exp_log_rnd_0, alpha_vec);
            const __m512 dq_val_vec_1 = _mm512_mul_ps(exp_log_rnd_1, alpha_vec);

            const __mmask16 threshold_mask_0 = _mm512_cmp_ps_mask(dq_val_vec_0, dq_threshold_vec_0, _CMP_LT_OQ);
            const __mmask16 threshold_mask_1 = _mm512_cmp_ps_mask(dq_val_vec_1, dq_threshold_vec_1, _CMP_LT_OQ);
            const __mmask16 zero_mask_0 = _mm512_cmp_ps_mask(val_vec_0, _mm512_zeros_ps, _CMP_EQ_OQ);
            const __mmask16 zero_mask_1 = _mm512_cmp_ps_mask(val_vec_1, _mm512_zeros_ps, _CMP_EQ_OQ);

            // Set -inf to sentinel values
            const __m512i log_trunc_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_0, log_abs_clipped_rcp_alpha_0);
            const __m512i log_trunc_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_mask_1, log_abs_clipped_rcp_alpha_1);
            const __m512i log_trunc_rnd_i0 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_0, log_abs_rnd_0);
            const __m512i log_trunc_rnd_i1 = _mm512_mask_cvttps_epi32(_mm512_127_epi32, neg_inf_rnd_mask_1, log_abs_rnd_1);

            const __m512i q_vec_tmp_0 = _mm512_mask_blend_epi32(threshold_mask_0, log_trunc_rnd_i0, log_trunc_i0);
            const __m512i quantized_vec_0 = _mm512_mask_blend_epi32(zero_mask_0, q_vec_tmp_0, _mm512_127_epi32);
            const __m512i q_vec_tmp_1 = _mm512_mask_blend_epi32(threshold_mask_1, log_trunc_rnd_i1, log_trunc_i1);
            const __m512i quantized_vec_1 = _mm512_mask_blend_epi32(zero_mask_1, q_vec_tmp_1, _mm512_127_epi32);

            // Pack and store the exponents
            // TODO: During shifting we have a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3
            // TODO: Try to pack them such that a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d0
            // TODO: During unpacking you can just shift the elements and use a mask to get rid of the extra elements on the left of the target num.
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

            int8_t q_val_1;
            if (val == 0.0f) {
                q_val_1 = 127;
            } else {
#ifdef QSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED
                const float alpha_rnd = 1.0f;
                const float rnd = 0.0f;
#else
                const float alpha_rnd = get_random_float();
                const float rnd = get_random_float() - 0.5f;
#endif
                // Compute
                const float alpha_eps = alpha * alpha_rnd;
                const float val_abs = fabs(val);
                const float sign = val < 0.0f ? -1.0f : 1.0f;
                const float clip_0 = val_abs < alpha ? alpha * sign : val;
                const float clip_1 = val_abs < alpha_eps ? 0.0f : clip_0;

                // Compute noise
                const float abs_clip = fabs(clip_1);
                const float abs_clip_mul_rcp_alpha = abs_clip * rcp_alpha;
                const float log_abs_clip_mul_rcp_alpha = log2f(abs_clip_mul_rcp_alpha); // TODO: Check for infinity
                const float log_trunc_f = floorf(log_abs_clip_mul_rcp_alpha);
                const float exp_log = exp2f(log_trunc_f);
                const float noise = exp_log * rnd;

                const float dq_threshold_val = exp_log * alpha;

                const float abs_clip_rnd_0 = abs_clip_mul_rcp_alpha + noise;
                const float abs_clip_rnd_1 = abs_clip_rnd_0 * 1.33333333333f;
                const float log_abs_rnd = log2f(abs_clip_rnd_1);
                const float log_rnd_trunc_f = floorf(log_abs_rnd);
                const float exp_log_rnd = exp2f(log_rnd_trunc_f);

                const float dq_val = exp_log_rnd * alpha;

                const int8_t log_trunc_i =
                        log_abs_clip_mul_rcp_alpha == NEG_INF ? 127 : (int8_t) log_abs_clip_mul_rcp_alpha;
                const int8_t log_trunc_rnd_i = log_abs_rnd == NEG_INF ? 127 : (int8_t) log_abs_rnd;

                q_val_1 = dq_val < dq_threshold_val ? log_trunc_i : log_trunc_rnd_i;
            }

            // Store
            group_result[idx] = q_val_1;
        }
    }
}

void quantization::LUQ8Strategy::restore(
    union Quantization_Input<int8_t>& input
) {
    const int8_t* values = input.luq_quantization_input.q_values;
    float* result = input.luq_quantization_input.dq_values;
    const float alpha = input.luq_quantization_input.scale;
    const __mmask16* signs = input.luq_quantization_input.signs;
    const int size = input.luq_quantization_input.size;
    const int mask_count = input.luq_quantization_input.mask_count;

    int idx;
    int mask_idx;
    const __m512 alpha_vec = _mm512_set1_ps(alpha);
    for (idx = 0, mask_idx = 0; idx < size - 63; idx += 64, mask_idx += 4) {
        // Load
        const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (values + idx));
        const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (values + idx + 32));
        const __mmask16 sign_mask_0 = signs[mask_idx];
        const __mmask16 sign_mask_1 = signs[mask_idx + 1];
        const __mmask16 sign_mask_2 = signs[mask_idx + 2];
        const __mmask16 sign_mask_3 = signs[mask_idx + 3];

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

        // TODO: This is probably too inefficient :)
        const __m512i i_pack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_0), unpacked_1, 1);
        const __m512i i_pack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_2), unpacked_3, 1);
        const __m512i i_pack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_4), unpacked_5, 1);
        const __m512i i_pack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_6), unpacked_7, 1);

        const __m512 f_pack_0_512 = _mm512_cvtepi32_ps(i_pack_0_512);
        const __m512 f_pack_1_512 = _mm512_cvtepi32_ps(i_pack_1_512);
        const __m512 f_pack_2_512 = _mm512_cvtepi32_ps(i_pack_2_512);
        const __m512 f_pack_3_512 = _mm512_cvtepi32_ps(i_pack_3_512);

        const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(f_pack_0_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(f_pack_1_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_2 = _mm512_cmp_ps_mask(f_pack_2_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_3 = _mm512_cmp_ps_mask(f_pack_3_512, _mm512_127_ps, _CMP_EQ_OQ);

        const __m512 f_inf_pack_0_512 = _mm512_mask_blend_ps(neg_inf_mask_0,  f_pack_0_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_1_512 = _mm512_mask_blend_ps(neg_inf_mask_1,  f_pack_1_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_2_512 = _mm512_mask_blend_ps(neg_inf_mask_2,  f_pack_2_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_3_512 = _mm512_mask_blend_ps(neg_inf_mask_3,  f_pack_3_512, _mm512_neg_inf_ps);

        const __m512 exp_f_pack_0_512 = _mm512_qsparse_pow2_ps(f_inf_pack_0_512);
        const __m512 exp_f_pack_1_512 = _mm512_qsparse_pow2_ps(f_inf_pack_1_512);
        const __m512 exp_f_pack_2_512 = _mm512_qsparse_pow2_ps(f_inf_pack_2_512);
        const __m512 exp_f_pack_3_512 = _mm512_qsparse_pow2_ps(f_inf_pack_3_512);

        // TODO: Can you make this more optimized?
        const __m512 alpha_exp_f_pack_0_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_0_512);
        const __m512 alpha_exp_f_pack_1_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_1_512);
        const __m512 alpha_exp_f_pack_2_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_2_512);
        const __m512 alpha_exp_f_pack_3_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_3_512);

        const __m512 s_alpha_exp_f_pack_0_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_0_512, sign_mask_0, alpha_exp_f_pack_0_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_1_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_1_512, sign_mask_1, alpha_exp_f_pack_1_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_2_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_2_512, sign_mask_2, alpha_exp_f_pack_2_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_3_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_3_512, sign_mask_3, alpha_exp_f_pack_3_512, _mm512_mones_ps);

        float* base = result + idx;
        _mm512_storeu_ps(base + 0, s_alpha_exp_f_pack_0_512);
        _mm512_storeu_ps(base + 16, s_alpha_exp_f_pack_1_512);
        _mm512_storeu_ps(base + 32, s_alpha_exp_f_pack_2_512);
        _mm512_storeu_ps(base + 48, s_alpha_exp_f_pack_3_512);
    }

    uint64_t mask = 0;
    for (int i = 0; i + mask_idx < mask_count; i++) {
        // Load
        const uint64_t sign_mask = signs[i + mask_idx];

        //Compute
        const int shift_amount =  16 * i;
        const uint64_t shifted_signs = sign_mask << shift_amount;
        const uint64_t concat_mask = mask | shifted_signs;

        // Store
        mask = concat_mask;
    }

    for (; idx < size; idx++) {
        // Load
        const int8_t q_val = values[idx];

        // Compute
        float dq_val;
        if (q_val == 127) {
            dq_val = 0.0f;
        } else {
            const float exp_val = exp2f(q_val);
            const float alpha_exp_val = alpha * exp_val;
            dq_val = (mask & 1ul) ? -alpha_exp_val : alpha_exp_val;
        }

        // Store
        result[idx] = dq_val;
        mask = mask >> 1;
    }
}

void quantization::LUQ8Strategy::restore_parallel(union Quantization_Input<int8_t>& input) {
#if defined(_OPENMP)
    const int8_t* values = input.luq_quantization_input.q_values;
    float* result = input.luq_quantization_input.dq_values;
    const float alpha = input.luq_quantization_input.scale;
    const __mmask16* signs = input.luq_quantization_input.signs;
    const int size = input.luq_quantization_input.size;
    const int mask_count = input.luq_quantization_input.mask_count;

    const int thread_count = get_OpenMP_threads();
    int* signIndices = new int[thread_count];
    int mask_incr = (size / 64 <= thread_count) ? 4 : (size * 4) / (64 * thread_count);
    for (int thread = 0; thread < thread_count; thread++) {
        signIndices[thread] = thread * mask_incr;
    }

    int idx;
    const __m512 alpha_vec = _mm512_set1_ps(alpha);
    #pragma omp parallel for default(none) shared(size, values, signs, signIndices, alpha_vec, result, _mm512_127_ps, _mm512_neg_inf_ps, _mm512_mones_ps)
    for (idx = 0; idx < size - 63; idx += 64) {
        const int tid = get_OpenMP_thread();

        // Load
        const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) (values + idx));
        const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (values + idx + 32));
        const __mmask16 sign_mask_0 = signs[signIndices[tid]];
        const __mmask16 sign_mask_1 = signs[signIndices[tid] + 1];
        const __mmask16 sign_mask_2 = signs[signIndices[tid] + 2];
        const __mmask16 sign_mask_3 = signs[signIndices[tid] + 3];
        signIndices[tid] += 4;

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

        // TODO: This is probably too inefficient :)
        const __m512i i_pack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_0), unpacked_1, 1);
        const __m512i i_pack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_2), unpacked_3, 1);
        const __m512i i_pack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_4), unpacked_5, 1);
        const __m512i i_pack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_6), unpacked_7, 1);

        const __m512 f_pack_0_512 = _mm512_cvtepi32_ps(i_pack_0_512);
        const __m512 f_pack_1_512 = _mm512_cvtepi32_ps(i_pack_1_512);
        const __m512 f_pack_2_512 = _mm512_cvtepi32_ps(i_pack_2_512);
        const __m512 f_pack_3_512 = _mm512_cvtepi32_ps(i_pack_3_512);

        const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(f_pack_0_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(f_pack_1_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_2 = _mm512_cmp_ps_mask(f_pack_2_512, _mm512_127_ps, _CMP_EQ_OQ);
        const __mmask16 neg_inf_mask_3 = _mm512_cmp_ps_mask(f_pack_3_512, _mm512_127_ps, _CMP_EQ_OQ);

        const __m512 f_inf_pack_0_512 = _mm512_mask_blend_ps(neg_inf_mask_0,  f_pack_0_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_1_512 = _mm512_mask_blend_ps(neg_inf_mask_1,  f_pack_1_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_2_512 = _mm512_mask_blend_ps(neg_inf_mask_2,  f_pack_2_512, _mm512_neg_inf_ps);
        const __m512 f_inf_pack_3_512 = _mm512_mask_blend_ps(neg_inf_mask_3,  f_pack_3_512, _mm512_neg_inf_ps);

        const __m512 exp_f_pack_0_512 = _mm512_qsparse_pow2_ps(f_inf_pack_0_512);
        const __m512 exp_f_pack_1_512 = _mm512_qsparse_pow2_ps(f_inf_pack_1_512);
        const __m512 exp_f_pack_2_512 = _mm512_qsparse_pow2_ps(f_inf_pack_2_512);
        const __m512 exp_f_pack_3_512 = _mm512_qsparse_pow2_ps(f_inf_pack_3_512);

        // TODO: Can you make this more optimized?
        const __m512 alpha_exp_f_pack_0_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_0_512);
        const __m512 alpha_exp_f_pack_1_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_1_512);
        const __m512 alpha_exp_f_pack_2_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_2_512);
        const __m512 alpha_exp_f_pack_3_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_3_512);

        const __m512 s_alpha_exp_f_pack_0_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_0_512, sign_mask_0, alpha_exp_f_pack_0_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_1_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_1_512, sign_mask_1, alpha_exp_f_pack_1_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_2_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_2_512, sign_mask_2, alpha_exp_f_pack_2_512, _mm512_mones_ps);
        const __m512 s_alpha_exp_f_pack_3_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_3_512, sign_mask_3, alpha_exp_f_pack_3_512, _mm512_mones_ps);

        float* base = result + idx;
        _mm512_storeu_ps(base + 0, s_alpha_exp_f_pack_0_512);
        _mm512_storeu_ps(base + 16, s_alpha_exp_f_pack_1_512);
        _mm512_storeu_ps(base + 32, s_alpha_exp_f_pack_2_512);
        _mm512_storeu_ps(base + 48, s_alpha_exp_f_pack_3_512);
    }

    uint64_t mask = 0;
    int mask_idx = (size / 64) * 4;
    for (int i = 0; i + mask_idx < mask_count; i++) {
        // Load
        const uint64_t sign_mask = signs[i + mask_idx];

        //Compute
        const int shift_amount =  16 * i;
        const uint64_t shifted_signs = sign_mask << shift_amount;
        const uint64_t concat_mask = mask | shifted_signs;

        // Store
        mask = concat_mask;
    }

    idx = (size / 64) * 64;
    for (; idx < size; idx++) {
        // Load
        const int8_t q_val = values[idx];

        // Compute
        float dq_val;
        if (q_val == 127) {
            dq_val = 0.0f;
        } else {
            const float exp_val = exp2f(q_val);
            const float alpha_exp_val = alpha * exp_val;
            dq_val = (mask & 1ul) ? -alpha_exp_val : alpha_exp_val;
        }

        // Store
        result[idx] = dq_val;
        mask = mask >> 1;
    }
#else
    quantization::LUQ8Strategy::restore(input);
#endif
}

void quantization::LUQ8Strategy::restore_grouped(union quantization::Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.luq_grouped_input.q_values;
    float* result = input.luq_grouped_input.dq_values;
    const float* alpha = input.luq_grouped_input.scale;
    const __mmask16* signs = input.luq_grouped_input.signs;
    const int size = input.luq_grouped_input.size;
    const int mask_count = input.luq_grouped_input.mask_count;

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
        const int group_mask_count = (group == group_count - 1)
             ? (group_size % qgroup_size) == 0
               ? (group_size >> 4)
               : (group_size + 15) / 16
             : (group_size >> 4);
        const int mask_start = group * (qgroup_size >> 4);
        const __mmask16* group_signs = signs + mask_start;

        int idx;
        int mask_idx;
        const __m512 alpha_vec = _mm512_set1_ps(alpha[group]);
        for (idx = 0, mask_idx = 0; idx < group_size - 63; idx += 64, mask_idx += 4) {
            // Load
            const int8_t* group_base = group_values + idx;
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) group_base);
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_base + 32));
            const __mmask16 sign_mask_0 = group_signs[mask_idx];
            const __mmask16 sign_mask_1 = group_signs[mask_idx + 1];
            const __mmask16 sign_mask_2 = group_signs[mask_idx + 2];
            const __mmask16 sign_mask_3 = group_signs[mask_idx + 3];

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

            // TODO: This is probably too inefficient :)
            const __m512i i_pack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_0), unpacked_1, 1);
            const __m512i i_pack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_2), unpacked_3, 1);
            const __m512i i_pack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_4), unpacked_5, 1);
            const __m512i i_pack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_6), unpacked_7, 1);

            const __m512 f_pack_0_512 = _mm512_cvtepi32_ps(i_pack_0_512);
            const __m512 f_pack_1_512 = _mm512_cvtepi32_ps(i_pack_1_512);
            const __m512 f_pack_2_512 = _mm512_cvtepi32_ps(i_pack_2_512);
            const __m512 f_pack_3_512 = _mm512_cvtepi32_ps(i_pack_3_512);

            const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(f_pack_0_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(f_pack_1_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_2 = _mm512_cmp_ps_mask(f_pack_2_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_3 = _mm512_cmp_ps_mask(f_pack_3_512, _mm512_127_ps, _CMP_EQ_OQ);

            const __m512 f_inf_pack_0_512 = _mm512_mask_blend_ps(neg_inf_mask_0,  f_pack_0_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_1_512 = _mm512_mask_blend_ps(neg_inf_mask_1,  f_pack_1_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_2_512 = _mm512_mask_blend_ps(neg_inf_mask_2,  f_pack_2_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_3_512 = _mm512_mask_blend_ps(neg_inf_mask_3,  f_pack_3_512, _mm512_neg_inf_ps);

            const __m512 exp_f_pack_0_512 = _mm512_qsparse_pow2_ps(f_inf_pack_0_512);
            const __m512 exp_f_pack_1_512 = _mm512_qsparse_pow2_ps(f_inf_pack_1_512);
            const __m512 exp_f_pack_2_512 = _mm512_qsparse_pow2_ps(f_inf_pack_2_512);
            const __m512 exp_f_pack_3_512 = _mm512_qsparse_pow2_ps(f_inf_pack_3_512);

            // TODO: Can you make this more optimized?
            const __m512 alpha_exp_f_pack_0_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_0_512);
            const __m512 alpha_exp_f_pack_1_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_1_512);
            const __m512 alpha_exp_f_pack_2_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_2_512);
            const __m512 alpha_exp_f_pack_3_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_3_512);

            const __m512 s_alpha_exp_f_pack_0_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_0_512, sign_mask_0, alpha_exp_f_pack_0_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_1_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_1_512, sign_mask_1, alpha_exp_f_pack_1_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_2_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_2_512, sign_mask_2, alpha_exp_f_pack_2_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_3_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_3_512, sign_mask_3, alpha_exp_f_pack_3_512, _mm512_mones_ps);

            float* base = group_result + idx;
            _mm512_storeu_ps(base + 0, s_alpha_exp_f_pack_0_512);
            _mm512_storeu_ps(base + 16, s_alpha_exp_f_pack_1_512);
            _mm512_storeu_ps(base + 32, s_alpha_exp_f_pack_2_512);
            _mm512_storeu_ps(base + 48, s_alpha_exp_f_pack_3_512);
        }

        uint64_t mask = 0;
        // TODO: Fix mask_count here
        for (int i = 0; i + mask_idx < group_mask_count; i++) {
            // Load
            const uint64_t sign_mask = group_signs[i + mask_idx];

            //Compute
            const int shift_amount =  16 * i;
            const uint64_t shifted_signs = sign_mask << shift_amount;
            const uint64_t concat_mask = mask | shifted_signs;

            // Store
            mask = concat_mask;
        }

        for (; idx < group_size; idx++) {
            // Load
            const int8_t q_val = group_values[idx];

            // Compute
            float dq_val;
            if (q_val == 127) {
                dq_val = 0.0f;
            } else {
                const float exp_val = exp2f(q_val);
                const float alpha_exp_val = alpha[group] * exp_val;
                dq_val = (mask & 1ul) ? -alpha_exp_val : alpha_exp_val;
            }

            // Store
            group_result[idx] = dq_val;
            mask = mask >> 1;
        }
    }
}

void quantization::LUQ8Strategy::restore_grouped_parallel(union Quantization_Input<int8_t>& input, int qgroup_size, int qgroup_shift_amount) {
    const int8_t* values = input.luq_grouped_input.q_values;
    float* result = input.luq_grouped_input.dq_values;
    const float* alpha = input.luq_grouped_input.scale;
    const __mmask16* signs = input.luq_grouped_input.signs;
    const int size = input.luq_grouped_input.size;
    const int mask_count = input.luq_grouped_input.mask_count;

    int group_count = (size + qgroup_size - 1) >> qgroup_shift_amount;
    #pragma omp parallel for default(none) shared(group_count, size, qgroup_size, values, result, _mm512_127_ps, _mm512_neg_inf_ps, _mm512_mones_ps, alpha, signs)
    for (int group = 0; group < group_count; group++) {
        const int group_size = (group == group_count - 1)
           ? (size % qgroup_size) == 0
             ? qgroup_size
             : size % qgroup_size
           : qgroup_size;
        const int group_start = group * qgroup_size;
        const int8_t* group_values = values + group_start;
        float* group_result = result + group_start;
        const int group_mask_count = (group == group_count - 1)
             ? (group_size % qgroup_size) == 0
               ? (group_size >> 4)
               : (group_size + 15) / 16
             : (group_size >> 4);
        const int mask_start = group * (qgroup_size >> 4);
        const __mmask16* group_signs = signs + mask_start;

        int idx;
        int mask_idx;
        const __m512 alpha_vec = _mm512_set1_ps(alpha[group]);
        for (idx = 0, mask_idx = 0; idx < group_size - 63; idx += 64, mask_idx += 4) {
            // Load
            const int8_t* group_base = group_values + idx;
            const __m256i q_val_vec_0 = _mm256_loadu_si256((__m256i *) group_base);
            const __m256i q_val_vec_1 = _mm256_loadu_si256((__m256i *) (group_base + 32));
            const __mmask16 sign_mask_0 = group_signs[mask_idx];
            const __mmask16 sign_mask_1 = group_signs[mask_idx + 1];
            const __mmask16 sign_mask_2 = group_signs[mask_idx + 2];
            const __mmask16 sign_mask_3 = group_signs[mask_idx + 3];

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

            // TODO: This is probably too inefficient :)
            const __m512i i_pack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_0), unpacked_1, 1);
            const __m512i i_pack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_2), unpacked_3, 1);
            const __m512i i_pack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_4), unpacked_5, 1);
            const __m512i i_pack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(unpacked_6), unpacked_7, 1);

            const __m512 f_pack_0_512 = _mm512_cvtepi32_ps(i_pack_0_512);
            const __m512 f_pack_1_512 = _mm512_cvtepi32_ps(i_pack_1_512);
            const __m512 f_pack_2_512 = _mm512_cvtepi32_ps(i_pack_2_512);
            const __m512 f_pack_3_512 = _mm512_cvtepi32_ps(i_pack_3_512);

            const __mmask16 neg_inf_mask_0 = _mm512_cmp_ps_mask(f_pack_0_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_1 = _mm512_cmp_ps_mask(f_pack_1_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_2 = _mm512_cmp_ps_mask(f_pack_2_512, _mm512_127_ps, _CMP_EQ_OQ);
            const __mmask16 neg_inf_mask_3 = _mm512_cmp_ps_mask(f_pack_3_512, _mm512_127_ps, _CMP_EQ_OQ);

            const __m512 f_inf_pack_0_512 = _mm512_mask_blend_ps(neg_inf_mask_0,  f_pack_0_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_1_512 = _mm512_mask_blend_ps(neg_inf_mask_1,  f_pack_1_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_2_512 = _mm512_mask_blend_ps(neg_inf_mask_2,  f_pack_2_512, _mm512_neg_inf_ps);
            const __m512 f_inf_pack_3_512 = _mm512_mask_blend_ps(neg_inf_mask_3,  f_pack_3_512, _mm512_neg_inf_ps);

            const __m512 exp_f_pack_0_512 = _mm512_qsparse_pow2_ps(f_inf_pack_0_512);
            const __m512 exp_f_pack_1_512 = _mm512_qsparse_pow2_ps(f_inf_pack_1_512);
            const __m512 exp_f_pack_2_512 = _mm512_qsparse_pow2_ps(f_inf_pack_2_512);
            const __m512 exp_f_pack_3_512 = _mm512_qsparse_pow2_ps(f_inf_pack_3_512);

            // TODO: Can you make this more optimized?
            const __m512 alpha_exp_f_pack_0_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_0_512);
            const __m512 alpha_exp_f_pack_1_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_1_512);
            const __m512 alpha_exp_f_pack_2_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_2_512);
            const __m512 alpha_exp_f_pack_3_512 = _mm512_mul_ps(alpha_vec, exp_f_pack_3_512);

            const __m512 s_alpha_exp_f_pack_0_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_0_512, sign_mask_0, alpha_exp_f_pack_0_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_1_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_1_512, sign_mask_1, alpha_exp_f_pack_1_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_2_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_2_512, sign_mask_2, alpha_exp_f_pack_2_512, _mm512_mones_ps);
            const __m512 s_alpha_exp_f_pack_3_512 = _mm512_mask_mul_ps(alpha_exp_f_pack_3_512, sign_mask_3, alpha_exp_f_pack_3_512, _mm512_mones_ps);

            float* base = group_result + idx;
            _mm512_storeu_ps(base + 0, s_alpha_exp_f_pack_0_512);
            _mm512_storeu_ps(base + 16, s_alpha_exp_f_pack_1_512);
            _mm512_storeu_ps(base + 32, s_alpha_exp_f_pack_2_512);
            _mm512_storeu_ps(base + 48, s_alpha_exp_f_pack_3_512);
        }

        uint64_t mask = 0;
        // TODO: Fix mask_count here
        for (int i = 0; i + mask_idx < group_mask_count; i++) {
            // Load
            const uint64_t sign_mask = group_signs[i + mask_idx];

            //Compute
            const int shift_amount =  16 * i;
            const uint64_t shifted_signs = sign_mask << shift_amount;
            const uint64_t concat_mask = mask | shifted_signs;

            // Store
            mask = concat_mask;
        }

        for (; idx < group_size; idx++) {
            // Load
            const int8_t q_val = group_values[idx];

            // Compute
            float dq_val;
            if (q_val == 127) {
                dq_val = 0.0f;
            } else {
                const float exp_val = exp2f(q_val);
                const float alpha_exp_val = alpha[group] * exp_val;
                dq_val = (mask & 1ul) ? -alpha_exp_val : alpha_exp_val;
            }

            // Store
            group_result[idx] = dq_val;
            mask = mask >> 1;
        }
    }
}

float quantization::LUQ8Strategy::get_threshold_denom() {
    return threshold_denom;
}