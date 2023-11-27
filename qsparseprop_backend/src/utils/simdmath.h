#ifndef QSPARSEPROP_SIMDMATH_H
#define QSPARSEPROP_SIMDMATH_H

#include <immintrin.h>
#include "src/conf/constants.h"

/**
 * Constants required to implement logarithm operation.
 * The code is taken from https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h
 * */
// AVX constants
const __m512 cephes_SQRTHF = _mm512_set1_ps(0.707106781186547524f);
const __m512 cephes_log_p0 = _mm512_set1_ps(7.0376836292E-2f);
const __m512 cephes_log_p1 = _mm512_set1_ps(-1.1514610310E-1f);
const __m512 cephes_log_p2 = _mm512_set1_ps(1.1676998740E-1f);
const __m512 cephes_log_p3 = _mm512_set1_ps(-1.2420140846E-1f);
const __m512 cephes_log_p4 = _mm512_set1_ps(1.4249322787E-1f);
const __m512 cephes_log_p5 = _mm512_set1_ps(-1.6668057665E-1f);
const __m512 cephes_log_p6 = _mm512_set1_ps(2.0000714765E-1f);
const __m512 cephes_log_p7 = _mm512_set1_ps(-2.4999993993E-1f);
const __m512 cephes_log_p8 = _mm512_set1_ps(3.3333331174E-1f);
const __m512 cephes_log_q1 = _mm512_set1_ps(-2.12194440e-4f);
const __m512 cephes_log_q2 = _mm512_set1_ps(0.693359375f);
const __m512 min_norm_pos = (__m512) _mm512_set1_epi32(0x00800000);
const __m512 inv_mant_mask = (__m512) _mm512_set1_epi32(~0x7f800000);
const __m512i _mm512_0x7f_epi = _mm512_set1_epi32(0x7f);
const __m512  log_2_ps = _mm512_set1_ps(0.69314718056f);
const __m512 rcp_log_2_ps = _mm512_set1_ps(1.44269504089f);

const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
const __m512 cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);
const __m512 cephes_exp_C1 = _mm512_set1_ps(-0.693359375f);
const __m512 cephes_exp_C2 = _mm512_set1_ps(2.12194440e-4f);
const __m512 cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
const __m512 cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
const __m512 cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
const __m512 cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
const __m512 cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
const __m512 cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

// hadd
float _mm512_haddf32_ss(__m512 vec);

// hadd 256-bit vector
float _mm256_haddf32_ss(__m256 acc);

// max
float _mm512_max_ss(float* values, int size);

// hmax
float _mm512_hmaxf32_ss(__m512 vec);

// min
float _mm512_min_ss(float* values, int size);

// hmin
float _mm512_hminf32_ss(__m512 vec);

// std
float _mm512_std_ss(float* values, int size);

// mean
float _mm512_mean_ss(float* values, int size);

// log
__m512 _mm512_qsparse_log_ps(__m512 vec);

// log2
__m512 _mm512_qsparse_log2_ps(__m512 vec);

// exp
__m512 _mm512_qsparse_exp_ps(__m512 vec);

// pow2
__m512 _mm512_qsparse_pow2_ps(__m512 vec);

#endif //QSPARSEPROP_SIMDMATH_H