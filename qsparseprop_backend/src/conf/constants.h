#ifndef QSPARSEPROP_CONSTANTS_H
#define QSPARSEPROP_CONSTANTS_H

#include <limits>
#include <immintrin.h>
#include <stdint.h>

// Scalar constants
const float MIN_FLOAT = -std::numeric_limits<float>::max();
const float MAX_FLOAT = std::numeric_limits<float>::max();
const float NEG_INF = -std::numeric_limits<float>::infinity();
const uint32_t qsparse_1st_bit_off_32  = 0x7FFFFFFFU;
const uint32_t qsparse_1st_bit_on_32 = 0x80000000U;
const float EPS = 1e-3f;

// Vector constants
const __m512i _mm512_1st_bit_off_epi8 = _mm512_set1_epi32(0x7f7f7f7f);
const __m512i _mm512_1st_bit_on_epi8 = _mm512_set1_epi32(0x80808080);
const __m512 _mm512_1st_bit_off = (__m512) _mm512_set1_epi32(qsparse_1st_bit_off_32);
const __m512 _mm512_1st_bit_on = (__m512) _mm512_set1_epi32(qsparse_1st_bit_on_32);
const __m512 _mm512_rcp_2pow31_ps = _mm512_set1_ps(1.0f / 2147483648.0f);
const __m512 _mm512_0_5_ps = _mm512_set1_ps(0.5f);
const __m512 _mm512_ones_ps = _mm512_set1_ps(1.0f);
const __m512 _mm512_mones_ps = _mm512_set1_ps(-1.0f);
const __m512 _mm512_zeros_ps = _mm512_setzero_ps();
const __m512i _mm512_zeros_epi32 = _mm512_setzero_si512();
const __m512 _mm512_full_mask_ps = (__m512) _mm512_set1_epi32(0xFFFFFFFF);
const __m512 _mm512_neg_inf_ps = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
const __m512 _mm512_4_div_3_ps = _mm512_set1_ps(4.0f / 3.0f);
const __m512 _mm512_127_ps = _mm512_set1_ps(127.0f);
const __m512i _mm512_127_epi32 = _mm512_set1_epi32(127);

const __m256i _mm256_8bit_perm_lo = _mm256_setr_epi8 (
    0, 4, 8, 12, 2, 6, 10, 14,
    1, 5, 9, 13, 3, 7, 11, 15,
    0, 4, 8, 12, 2, 6, 10, 14,
    1, 5, 9, 13, 3, 7, 11, 15
);
const __m256i _mm256_8bit_perm_hi = _mm256_setr_epi8 (
    2, 6, 10, 14, 0, 4, 8, 12,
    3, 7, 11, 15, 1, 5, 9, 13,
    2, 6, 10, 14, 0, 4, 8, 12,
    3, 7, 11, 15, 1, 5, 9, 13
);

const __m256i __mm512_8bit_restore_perm_lo = _mm256_setr_epi8(
    0, 8, -128, -128, 1, 9, -128, -128,
    2, 10, -128, -128, 3, 11, -128, -128,
    -128, -128, 4, 12, -128, -128, 5, 13,
    -128, -128, 6, 14, -128, -128, 7, 15
);
const __m256i __mm512_8bit_restore_perm_hi = _mm256_setr_epi8(
    -128, -128, 0, 8, -128, -128, 1, 9,
    -128, -128, 2, 10, -128, -128, 3, 11,
    4, 12, -128, -128, 5, 13, -128, -128,
    6, 14, -128, -128, 7, 15, -128, -128
);

const __m128i __mm_8bit_restore_shuffle_lo = _mm_set_epi8(
    3, -128, -128, -128, 2, -128, -128, -128,
    1,-128, -128, -128, 0, -128, -128, -128
);

const __m128i __mm_8bit_restore_shuffle_hi = _mm_set_epi8(
    7, -128, -128, -128, 6, -128, -128, -128,
    5,-128, -128, -128, 4, -128, -128, -128
);

#endif //QSPARSEPROP_CONSTANTS_H
