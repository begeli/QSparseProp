#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/quantization/dithered_quantization8_strategy.h"

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectly) {
    float arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        -57, -55, -53, -52, -50, -48, -46, -44,
        -43, -41, -39, -37, -36, -34, -32, -30,
        -28, -27, -25, -23, -21, -20, -18, -16,
        -14, -12, -11, -9, -7, -5, -4, -2,
        0, 2, 4, 5, 7, 9, 11, 12,
        14, 16, 18, 20, 21, 23, 25, 27,
        28, 30, 32, 34, 36, 37, 39, 41,
        43, 44, 46, 48, 50, 52, 53, 55,
        57
    };
    union quantization::Quantization_Input<int8_t> input
            = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.5628f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyGrouped) {
    float arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        -68, -66, -64, -62, -60, -57, -55, -53,
        -51, -49, -47, -44, -42, -40, -38, -36,
        -34, -31, -29, -27, -25, -23, -21, -18,
        -16, -14, -12, -10, -8, -5, -3, -1,
        0, 2, 4, 6, 9, 11, 13, 15,
        17, 19, 22, 24, 26, 28, 30, 32,
        35, 37, 39, 41, 43, 45, 48, 50,
        52, 54, 56, 58, 61, 63, 65, 67,
        1
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.461655f, 0.461655f, 32.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    float expected_dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.05f);

    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dequantization_const[i], 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyGroupedParallel) {
    float arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        -68, -66, -64, -62, -60, -57, -55, -53,
        -51, -49, -47, -44, -42, -40, -38, -36,
        -34, -31, -29, -27, -25, -23, -21, -18,
        -16, -14, -12, -10, -8, -5, -3, -1,
        0, 2, 4, 6, 9, 11, 13, 15,
        17, 19, 22, 24, 26, 28, 30, 32,
        35, 37, 39, 41, 43, 45, 48, 50,
        52, 54, 56, 58, 61, 63, 65, 67,
        1
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.461655f, 0.461655f, 32.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    float expected_dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.05f);

    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dequantization_const[i], 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyParallel) {
    float arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        -57, -55, -53, -52, -50, -48, -46, -44,
        -43, -41, -39, -37, -36, -34, -32, -30,
        -28, -27, -25, -23, -21, -20, -18, -16,
        -14, -12, -11, -9, -7, -5, -4, -2,
        0, 2, 4, 5, 7, 9, 11, 12,
        14, 16, 18, 20, 21, 23, 25, 27,
        28, 30, 32, 34, 36, 37, 39, 41,
        43, 44, 46, 48, 50, 52, 53, 55,
        57
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.5628f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectly) {
    int8_t q_arr[65] = {
        -56, -54, -52, -51, -49, -47, -45, -43,
        -42, -40, -38, -36, -35, -33, -31, -29,
        -27, -26, -24, -22, -20, -19, -17, -15,
        -13, -11, -10, -8, -6, -4, -3, -1,
        0, 2, 4, 5, 7, 9, 11, 12,
        14, 16, 18, 20, 21, 23, 25, 27,
        28, 30, 32, 34, 36, 37, 39, 41,
        43, 44, 46, 48, 50, 52, 53, 55,
        57
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    float scale = 0.5628f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 1.0f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyParallel) {
    int8_t q_arr[65] = {
        -56, -54, -52, -51, -49, -47, -45, -43,
        -42, -40, -38, -36, -35, -33, -31, -29,
        -27, -26, -24, -22, -20, -19, -17, -15,
        -13, -11, -10, -8, -6, -4, -3, -1,
        0, 2, 4, 5, 7, 9, 11, 12,
        14, 16, 18, 20, 21, 23, 25, 27,
        28, 30, 32, 34, 36, 37, 39, 41,
        43, 44, 46, 48, 50, 52, 53, 55,
        57
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    float scale = 0.5628f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 1.0f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyGrouped) {
    int8_t q_arr[65] = {
        -68, -66, -64, -62, -60, -57, -55, -53,
        -51, -49, -47, -44, -42, -40, -38, -36,
        -34, -31, -29, -27, -25, -23, -21, -18,
        -16, -14, -12, -10, -8, -5, -3, -1,
        0, 2, 4, 6, 9, 11, 13, 15,
        17, 19, 22, 24, 26, 28, 30, 32,
        35, 37, 39, 41, 43, 45, 48, 50,
        52, 54, 56, 58, 61, 63, 65, 67,
        1
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    float scale[3] = {0.461655f, 0.461655f, 32.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.05f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 1.0f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyGroupedParallel) {
    int8_t q_arr[65] = {
        -68, -66, -64, -62, -60, -57, -55, -53,
        -51, -49, -47, -44, -42, -40, -38, -36,
        -34, -31, -29, -27, -25, -23, -21, -18,
        -16, -14, -12, -10, -8, -5, -3, -1,
        0, 2, 4, 6, 9, 11, 13, 15,
        17, 19, 22, 24, 26, 28, 30, 32,
        35, 37, 39, 41, 43, 45, 48, 50,
        52, 54, 56, 58, 61, 63, 65, 67,
        1
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0,17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0
    };
    float scale[3] = {0.461655f, 0.461655f, 32.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.05f);
    strategy.restore_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 1.0f));
    }
}

/*EST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllEq) {
    float arr[65] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(5.0f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}*/

/*TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllEqParallel) {
    float arr[65] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(5.0f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}*/

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllEqGrouped) {
    float arr[65] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float scale[3] = {5.0f, 5.0f, 5.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(5.0f, 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllEqGroupedParallel) {
    float arr[65] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float scale[3] = {5.0f, 5.0f, 5.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(5.0f, 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllEq) {
    int8_t q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float dq_arr[65];
    float scale = 5.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(5.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllEqParallel) {
    int8_t q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float dq_arr[65];
    float scale = 5.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(5.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllEqGrouped) {
    int8_t q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float dq_arr[65];
    float scale[3] = {5.0f, 5.0f, 5.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(5.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllEqGroupedParallel) {
    int8_t q_arr[65] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1
    };
    float dq_arr[65];
    float scale[3] = {5.0f, 5.0f, 5.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(5.0f, 0.1f));
    }
}

/*TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllZero) {
    float arr[65] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0
    };
    int8_t q_arr[65];
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.0f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}*/

/*TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllZeroParallel) {
    float arr[65] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0
    };
    int8_t q_arr[65];
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.0f, 0.1f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(0.0f, 0.1f));
}*/

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllZeroParallelGrouped) {
    float arr[65] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0
    };
    int8_t q_arr[65];
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(0.0f, 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestQuantizeCorrectlyAllZeroParallelGroupedParallel) {
    float arr[65] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0
    };
    int8_t q_arr[65];
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dequantization_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dequantization_const}};
    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(0.0f, 0.1f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllZero) {
    int8_t q_arr[65] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0
    };
    float dq_arr[65];
    float scale = 0.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.000001f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllZeroParallel) {
    int8_t q_arr[65] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0
    };
    float dq_arr[65];
    float scale = 0.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.000001f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllZeroGrouped) {
    int8_t q_arr[65] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0
    };
    float dq_arr[65];
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.000001f));
    }
}

TEST(Test_Dithered_Quantization_8, TestRestoreCorrectlyAllZeroGroupedParallel) {
    int8_t q_arr[65] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0
    };
    float dq_arr[65];
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::DitheredQuantization8Strategy strategy
        = quantization::DitheredQuantization8Strategy(0.03f);
    strategy.restore_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.000001f));
    }
}
