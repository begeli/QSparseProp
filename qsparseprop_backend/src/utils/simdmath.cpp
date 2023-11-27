#include "simdmath.h"

float _mm512_haddf32_ss(__m512 vec) {
    const __m256 tmp1 = _mm512_castps512_ps256(vec);
    const __m256 tmp2 = _mm512_extractf32x8_ps(vec, 1);
    const __m256 tmp3 = _mm256_add_ps(tmp1, tmp2);
    const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
    const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
    const __m128 tmp6 = _mm_add_ps(tmp4, tmp5);
    const __m128 tmp7 = _mm_add_ps(tmp6, _mm_movehl_ps(tmp6, tmp6));
    const __m128 tmp8 = _mm_add_ss(tmp7, _mm_shuffle_ps(tmp7, tmp7, 0x55));

    return _mm_cvtss_f32(tmp8);
}

float _mm256_haddf32_ss(__m256 acc) {
    const __m128 left  = _mm256_extractf128_ps(acc, 1);
    const __m128 right = _mm256_castps256_ps128(acc);
    const __m128 x128  = _mm_add_ps(left, right);
    const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return  _mm_cvtss_f32(x32);
}

// TODO: Check if there is a more efficient solution using permutes
float _mm512_hmaxf32_ss(__m512 vec) {
    const __m256 tmp1 = _mm512_castps512_ps256(vec);
    const __m256 tmp2 = _mm512_extractf32x8_ps(vec, 1);
    const __m256 tmp3 = _mm256_max_ps(tmp1, tmp2);
    const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
    const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
    const __m128 tmp6 = _mm_max_ps(tmp4, tmp5);
    const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
    const __m128 tmp8 = _mm_max_ps(tmp6, tmp7);
    const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
    const __m128 tmp0 = _mm_max_ps(tmp8, tmp9);

    // Return the result stored in the first element
    return _mm_cvtss_f32(tmp0);
}

float _mm512_hminf32_ss(__m512 vec) {
    const __m256 tmp1 = _mm512_castps512_ps256(vec);
    const __m256 tmp2 = _mm512_extractf32x8_ps(vec, 1);
    const __m256 tmp3 = _mm256_min_ps(tmp1, tmp2);
    const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
    const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
    const __m128 tmp6 = _mm_min_ps(tmp4, tmp5);
    const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
    const __m128 tmp8 = _mm_min_ps(tmp6, tmp7);
    const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
    const __m128 tmp0 = _mm_min_ps(tmp8, tmp9);

    // Return the result stored in the first element
    return _mm_cvtss_f32(tmp0);
}

__m512 _mm512_qsparse_log_ps(__m512 vec) {
    const __mmask16 invalid_mask = _mm512_cmp_ps_mask(vec, _mm512_zeros_ps, _CMP_LT_OS);
    const __mmask16 inf_mask = _mm512_cmp_ps_mask(vec, _mm512_zeros_ps, _CMP_EQ_OS);

    const __m512 vec_0 = _mm512_max_ps(vec, min_norm_pos);  /* cut off denormalized stuff */

    const __m512i imm0_0 = _mm512_srli_epi32(_mm512_castps_si512(vec_0), 23);

    /* keep only the fractional part */
    const __m512 vec_1 = _mm512_and_ps(vec_0, inv_mant_mask);
    const __m512 vec_2 = _mm512_or_ps(vec_1, _mm512_0_5_ps);

    const __m512i imm0_1 = _mm512_sub_epi32(imm0_0, _mm512_0x7f_epi);
    const __m512 e_0 = _mm512_cvtepi32_ps(imm0_1);

    const __m512 e_1 = _mm512_add_ps(e_0, _mm512_ones_ps);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    const __mmask16 mask = _mm512_cmp_ps_mask(vec_2, cephes_SQRTHF, _CMP_LT_OS);
    const __m512 tmp_0 = _mm512_mask_and_ps(_mm512_zeros_ps, mask, vec_2, vec_2);//_mm512_and_ps(vec, mask);
    const __m512 vec_3 = _mm512_sub_ps(vec_2, _mm512_ones_ps);
    const __m512 e_2 = _mm512_sub_ps(e_1, _mm512_mask_and_ps(_mm512_zeros_ps, mask, _mm512_ones_ps, _mm512_ones_ps));
    const __m512 vec_4 = _mm512_add_ps(vec_3, tmp_0);

    const __m512 log_tmp_0 = cephes_log_p0;
    const __m512 log_tmp_1 = _mm512_fmadd_ps(log_tmp_0, vec_4, cephes_log_p1);
    const __m512 log_tmp_2 = _mm512_fmadd_ps(log_tmp_1, vec_4, cephes_log_p2);
    const __m512 log_tmp_3 = _mm512_fmadd_ps(log_tmp_2, vec_4, cephes_log_p3);
    const __m512 log_tmp_4 = _mm512_fmadd_ps(log_tmp_3, vec_4, cephes_log_p4);
    const __m512 log_tmp_5 = _mm512_fmadd_ps(log_tmp_4, vec_4, cephes_log_p5);
    const __m512 log_tmp_6 = _mm512_fmadd_ps(log_tmp_5, vec_4, cephes_log_p6);
    const __m512 log_tmp_7 = _mm512_fmadd_ps(log_tmp_6, vec_4, cephes_log_p7);
    const __m512 log_tmp_8 = _mm512_fmadd_ps(log_tmp_7, vec_4, cephes_log_p8);
    const __m512 log_tmp_9 = _mm512_mul_ps(log_tmp_8, vec_4);

    const __m512 log_tmp_10 = _mm512_mul_ps(vec_4,vec_4);
    const __m512 log_tmp_11 = _mm512_mul_ps(log_tmp_9, log_tmp_10);
    const __m512 log_tmp_12 = _mm512_fmadd_ps(e_2, cephes_log_q1, log_tmp_11);
    const __m512 log_tmp_13 = _mm512_fmsub_ps(log_tmp_10, _mm512_0_5_ps, log_tmp_12);

    const __m512 vec_5 = _mm512_add_ps(vec_4, log_tmp_13);
    const __m512 vec_6 = _mm512_fmadd_ps(e_2, cephes_log_q2, vec_5);
    const __m512 vec_7 = _mm512_mask_or_ps(vec_6, invalid_mask, _mm512_full_mask_ps, _mm512_full_mask_ps); // negative arg will be NAN
    const __m512 vec_8 = _mm512_mask_or_ps(vec_7, inf_mask, _mm512_neg_inf_ps, _mm512_neg_inf_ps); // 0 will be -inf

    return vec_8;
}

__m512 _mm512_qsparse_log2_ps(__m512 vec) {
    const __m512 log_vec = _mm512_qsparse_log_ps(vec);
    return _mm512_mul_ps(log_vec, rcp_log_2_ps);
}

__m512 _mm512_qsparse_exp_ps(__m512 vec) {
    const __m512 vec_0 = _mm512_min_ps(vec, exp_hi);
    const __m512 vec_1 = _mm512_max_ps(vec_0, exp_lo);

    /* express exp(vec) as exp(g + n*log(2)) */
    const __m512 fx_0 = _mm512_fmadd_ps(vec_1, cephes_LOG2EF, _mm512_0_5_ps);

    /* how to perform a floorf with SSE: just below */
    const __m512 tmp_0 = _mm512_floor_ps(fx_0);

    /* if greater, substract 1 */
    const __mmask16 mask_0 = _mm512_cmp_ps_mask(tmp_0, fx_0, _CMP_GT_OS);
    const __m512 mask_1 = _mm512_mask_and_ps(_mm512_zeros_ps, mask_0, _mm512_ones_ps, _mm512_ones_ps);
    const __m512 fx_1 = _mm512_sub_ps(tmp_0, mask_1);

    const __m512 vec_2 = _mm512_fmadd_ps(fx_1, cephes_exp_C1, vec_1);
    const __m512 vec_3 = _mm512_fmadd_ps(fx_1, cephes_exp_C2, vec_2);

    const __m512 exp_tmp_1 = _mm512_mul_ps(vec_3, vec_3);

    const __m512 exp_tmp_2 = cephes_exp_p0;
    const __m512 exp_tmp_3 = _mm512_fmadd_ps(exp_tmp_2, vec_3, cephes_exp_p1);
    const __m512 exp_tmp_4 = _mm512_fmadd_ps(exp_tmp_3, vec_3, cephes_exp_p2);
    const __m512 exp_tmp_5 = _mm512_fmadd_ps(exp_tmp_4, vec_3, cephes_exp_p3);
    const __m512 exp_tmp_6 = _mm512_fmadd_ps(exp_tmp_5, vec_3, cephes_exp_p4);
    const __m512 exp_tmp_7 = _mm512_fmadd_ps(exp_tmp_6, vec_3, cephes_exp_p5);
    const __m512 exp_tmp_8 = _mm512_fmadd_ps(exp_tmp_7, exp_tmp_1, vec_3);
    const __m512 exp_tmp_9 = _mm512_add_ps(exp_tmp_8, _mm512_ones_ps);

    /* build 2^n */
    const __m512i imm0_0 = _mm512_cvttps_epi32(fx_1);
    // another two AVX2 instructions
    const __m512i imm0_1 = _mm512_add_epi32(imm0_0, _mm512_0x7f_epi);
    const __m512i imm0_2 = _mm512_slli_epi32(imm0_1, 23);
    const __m512 pow2n = _mm512_castsi512_ps(imm0_2);
    const __m512 exp = _mm512_mul_ps(exp_tmp_9, pow2n);
    return exp;
}

__m512 _mm512_qsparse_pow2_ps(__m512 vec) {
    const __m512 exponent = _mm512_mul_ps(vec, log_2_ps);
    return _mm512_qsparse_exp_ps(exponent);
}