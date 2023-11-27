#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/utils/simdmath.h"
#include <cmath>

TEST(Test_SIMD_Math, Test_Hadd) {
    __m512 vec = _mm512_set1_ps(1.0f);
    float sum = _mm512_haddf32_ss(vec);
    ASSERT_THAT(sum, testing::FloatNear(16.0f, 0.001f));
}

TEST(Test_SIMD_Math, Test_Hmin) {
    __m512 vec = _mm512_set_ps(
        -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    );
    float hmin = _mm512_hminf32_ss(vec);
    EXPECT_EQ(hmin, -1.0f);
}

TEST(Test_SIMD_Math, Test_Hmax) {
    __m512 vec = _mm512_set_ps(
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    );
    float hmax = _mm512_hmaxf32_ss(vec);
    EXPECT_EQ(hmax, 1.0f);
}

TEST(Test_SIMD_Math, Test_Log) {
    float arr[16];

    __m512 ones = _mm512_set1_ps(1.0f);
    __m512 res = _mm512_qsparse_log_ps(ones);
    _mm512_storeu_ps(arr, res);
    ASSERT_THAT(arr[0], testing::FloatNear(0.0f, 0.1f));

    __m512 eulers_num = _mm512_set1_ps(2.71828);
    res = _mm512_qsparse_log_ps(eulers_num);
    _mm512_storeu_ps(arr, res);
    ASSERT_THAT(arr[0], testing::FloatNear(1.0f, 0.2f));
}

TEST(Test_SIMD_Math, Test_Log2) {
    float arr[16];

    __m512 ones = _mm512_set1_ps(1.0f);
    __m512 res = _mm512_qsparse_log2_ps(ones);
    _mm512_storeu_ps(arr, res);
    ASSERT_THAT(arr[0], testing::FloatNear(0.0f, 0.1f));

    // TODO: There might be a bug for i >= -127
    for (int i = -126; i <= 127; i++) {
        __m512 vec = _mm512_set1_ps(powf(2.0f, (float) i));
        res = _mm512_qsparse_log2_ps(vec);
        _mm512_storeu_ps(arr, res);
        ASSERT_THAT(arr[0], testing::FloatNear((float) i, 0.1f));
    }
}

TEST(Test_SIMD_Math, Test_Exp) {
    // Chose these ranges, o.w the exponents overflow
    for (int i = -87; i <= 87; i++) {
        float arr[16];
        __m512 vec = _mm512_set1_ps((float) i);
        _mm512_storeu_ps(arr, _mm512_qsparse_exp_ps(vec));
        ASSERT_THAT(arr[0], testing::FloatNear(exp(i), exp(i) * 0.01));
    }
}

TEST(Test_SIMD_Math, Test_Pow2) {
    for (int i = -126; i <= 127; i++) {
        float arr[16];
        __m512 vec = _mm512_set1_ps((float) i);
        _mm512_storeu_ps(arr, _mm512_qsparse_pow2_ps(vec));
        ASSERT_THAT(arr[0], testing::FloatNear(powf(2.0f, i), powf(2.0f, i) * 0.01));
    }
}