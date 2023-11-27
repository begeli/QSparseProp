#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/quantization/luq8_strategy.h"

TEST(Test_LUQ_8, TestInitLUQStrategyCorrectly) {
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);
    ASSERT_THAT(strategy.get_threshold_denom(), testing::FloatNear(3.0517578125e-05, 3.0517578125e-06));
}

TEST(Test_LUQ_8, TestQuantizeCorrectly) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7,
        7, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 4, 4, 4, 3, 2,
        127, 2, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        2, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        2
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize(input);
    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(8192.0f * strategy.get_threshold_denom(), 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyParallel) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7,
        7, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 4, 4, 4, 3, 2,
        127, 2, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        2, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        2
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_parallel(input);
    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(8192.0f * strategy.get_threshold_denom(), 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyGrouped) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[6];
    float expected_scale[6] = {0.25, 0.000488281, 0.000457764, 0.000976562, 0.000976562, 3.05176e-05};
    float dq_const[6];
    int8_t expected_q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
        15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped(input, 16, 4);

    for (int i = 0; i < 81; i++) {
        //std::cout << (int) q_arr[i] << std::endl;
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 6; i++) {
        ASSERT_THAT(scale[i], testing::FloatNear(expected_scale[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyGroupedParallel) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[6];
    float expected_scale[6] = {0.25, 0.000488281, 0.000457764, 0.000976562, 0.000976562, 3.05176e-05};
    float dq_const[6];
    int8_t expected_q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
        15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped_parallel(input, 16, 4);

    for (int i = 0; i < 81; i++) {
        //std::cout << (int) q_arr[i] << std::endl;
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 6; i++) {
        ASSERT_THAT(scale[i], testing::FloatNear(expected_scale[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectly) {
    float arr[81];
    int8_t q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7,
        7, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 4, 4, 4, 3, 2,
        127, 2, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        2, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        2
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale = 8192.0f * 3.0517578125e-05;
    float dq_const;
    float expected_dq_arr[81] = {
        -8192, -32, -32, -32, -32, -32, -32, -32,
        -32, -16, -16, -16, -16, -16, -16, -16,
        -16, -16, -16, -16, -16, -8, -8, -8,
        -8, -8, -8, -4, -4, -4, -2, -1,
        0, 1, 2, 4, 4, -4, 8, 8,
        -8, 8, 8, 8, 16, 16, 16, 16,
        -16, 16, -16, 16, 16, 16, 16, 16,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore(input);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyParallel) {
    float arr[81];
    int8_t q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7,
        7, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 4, 4, 4, 3, 2,
        127, 2, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        2, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        2
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale = 8192.0f * 3.0517578125e-05;
    float dq_const;
    float expected_dq_arr[81] = {
        -8192, -32, -32, -32, -32, -32, -32, -32,
        -32, -16, -16, -16, -16, -16, -16, -16,
        -16, -16, -16, -16, -16, -8, -8, -8,
        -8, -8, -8, -4, -4, -4, -2, -1,
        0, 1, 2, 4, 4, -4, 8, 8,
        -8, 8, 8, 8, 16, 16, 16, 16,
        -16, 16, -16, 16, 16, 16, 16, 16,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_parallel(input);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyGrouped) {
    float arr[81];
    int8_t q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
        15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale[6] = {0.25, 0.000488281, 0.000457764, 0.000976562, 0.000976562, 3.05176e-05};
    float dq_const[6];
    float expected_dq_arr[81] = {
        -8192, -32, -32, -32, -32, -32, -32, -32,
        -32, -16, -16, -16, -16, -16, -16, -16,
        -16, -16, -16, -16, -16,  -8,  -8,  -8,
        -8,  -8,  -8,  -4, -4,  -4,  -2,  -1,
        0.0000, 0.9375, 1.8750, 3.7500, 3.7500, -3.7500, 7.5000, 7.5000,
        -7.5000, 7.5000, 7.5000, 7.5000, 15.0000, 15.0000, 15.0000,15.0000,
        -16.0, 16.0, -16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
        32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyGroupedParallel) {
    float arr[81];
    int8_t q_arr[81] = {
        15, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
        15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale[6] = {0.25, 0.000488281, 0.000457764, 0.000976562, 0.000976562, 3.05176e-05};
    float dq_const[6];
    float expected_dq_arr[81] = {
        -8192, -32, -32, -32, -32, -32, -32, -32,
        -32, -16, -16, -16, -16, -16, -16, -16,
        -16, -16, -16, -16, -16,  -8,  -8,  -8,
        -8,  -8,  -8,  -4, -4,  -4,  -2,  -1,
        0.0000, 0.9375, 1.8750, 3.7500, 3.7500, -3.7500, 7.5000, 7.5000,
        -7.5000, 7.5000, 7.5000, 7.5000, 15.0000, 15.0000, 15.0000,15.0000,
        -16.0, 16.0, -16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
        32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyAllZero) {
    float arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    int8_t q_arr[33];
    int mask_count = (33 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[3] = {
        0, 0, 0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize(input);
    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(0.0f, 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyAllZeroParallel) {
    float arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    int8_t q_arr[33];
    int mask_count = (33 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[3] = {
        0, 0, 0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_parallel(input);
    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(0.0f, 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyAllZeroGrouped) {
    float arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    int8_t q_arr[33];
    int mask_count = (33 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[3];
    float expected_scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3];
    int8_t expected_q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[3] = {
        0, 0, 0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(input.luq_grouped_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_grouped_input.signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.luq_grouped_input.scale[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyAllZeroGroupedParallel) {
    float arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    int8_t q_arr[33];
    int mask_count = (33 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[3];
    float expected_scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3];
    int8_t expected_q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[3] = {
        0, 0, 0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(input.luq_grouped_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_grouped_input.signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.luq_grouped_input.scale[i], testing::FloatNear(0.0f, 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyAllZero) {
    float arr[33];
    int8_t q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 3;
    __mmask16 signs[3] = {
        0, 0, 0
    };
    float scale = 0.0f;
    float dq_const;
    float expected_dq_arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyAllZeroParallel) {
    float arr[33];
    int8_t q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 3;
    __mmask16 signs[3] = {
        0, 0, 0
    };
    float scale = 0.0f;
    float dq_const;
    float expected_dq_arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_parallel(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyAllZeroGrouped) {
    float arr[33];
    int8_t q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 3;
    __mmask16 signs[3] = {
        0, 0, 0
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3];
    float expected_dq_arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyAllZeroGroupedParallel) {
    float arr[33];
    int8_t q_arr[33] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 3;
    __mmask16 signs[3] = {
        0, 0, 0
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float dq_const[3];
    float expected_dq_arr[33] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 33, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyLargeMax) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -1e38, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 1e38,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize(input);
    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(1e38 * strategy.get_threshold_denom(), 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyLargeMaxParallel) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -1e38, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 1e38,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale;
    float dq_const;
    int8_t expected_q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_parallel(input);
    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_quantization_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_quantization_input.signs[i], expected_signs[i]);
    }
    ASSERT_THAT(input.luq_quantization_input.scale, testing::FloatNear(1e38 * strategy.get_threshold_denom(), 0.1f));
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyLargeMaxGrouped) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -1e38, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 1e38,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[6];
    float expected_scale[6] = {3.05176e+33, 0.000488281, 0.000457764, 3.05176e+33, 0.000976562, 3.05176e-05};
    float dq_const[6];
    int8_t expected_q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        15, 15, 15, 15, 15, 14, 14, 14,
        14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 15, 15, 15, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        10, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped(input, 16, 4);

    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_grouped_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_grouped_input.signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 6; i++) {
        ASSERT_THAT(input.luq_grouped_input.scale[i], testing::FloatNear(expected_scale[i], expected_scale[i] * 0.001f));
    }
}

TEST(Test_LUQ_8, TestQuantizeCorrectlyLargeMaxGroupedParallel) {
    float arr[81] = {
        -8192.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -1e38, -20.0, -19.0, -18.0, -17.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        -8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        -16.0, 17.0, -18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 1e38,
        -1.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0,
        -1.0
    };
    int8_t q_arr[81];
    int mask_count = (81 + 15) / 16;
    __mmask16 signs[mask_count];
    float scale[6];
    float expected_scale[6] = {3.05176e+33, 0.000488281, 0.000457764, 3.05176e+33, 0.000976562, 3.05176e-05};
    float dq_const[6];
    int8_t expected_q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        15, 15, 15, 15, 15, 14, 14, 14,
        14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 15, 15, 15, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        10, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    __mmask16 expected_signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.quantize_grouped_parallel(input, 16, 4);

    for (int i = 0; i < 81; i++) {
        EXPECT_EQ(input.luq_grouped_input.q_values[i], expected_q_arr[i]);
    }

    for (int i = 0; i < mask_count; i++) {
        EXPECT_EQ(input.luq_grouped_input.signs[i], expected_signs[i]);
    }

    for (int i = 0; i < 6; i++) {
        ASSERT_THAT(input.luq_grouped_input.scale[i], testing::FloatNear(expected_scale[i], expected_scale[i] * 0.001f));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyLargeMax) {
    float arr[81];
    int8_t q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale = 1e38 * 3.0517578125e-05;
    float dq_const;
    float expected_dq_arr[81] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, -1e38, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1e38,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore(input);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], fabs(expected_dq_arr[i] * 0.01f)));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyLargeMaxParallel) {
    float arr[81];
    int8_t q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale = 1e38 * 3.0517578125e-05;
    float dq_const;
    float expected_dq_arr[81] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, -1e38, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1e38,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_quantization_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_parallel(input);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_quantization_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], fabs(expected_dq_arr[i] * 0.01f)));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyLargeMaxGrouped) {
    float arr[81];
    int8_t q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        15, 15, 15, 15, 15, 14, 14, 14,
        14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 15, 15, 15, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        10, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale[6] = {3.05176e+33, 0.000488281, 0.000457764, 3.05176e+33, 0.000976562, 3.05176e-05};
    float dq_const[6];
    float expected_dq_arr[81] = {
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, -1.0000e+38, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        -16, -16, -16, -16, -16, -8, -8, -8,
        -8, -8, -8, -4, -4, -4, -2, -1,
        0.0000, 0.9375, 1.8750, 3.7500, 3.7500, -3.7500, 7.5000, 7.5000,
        -7.5000, 7.5000, 7.5000, 7.5000, 15.0000, 15.0000, 15.0000, 15.0000,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+38,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], fabs(expected_dq_arr[i] * 0.01f)));
    }
}

TEST(Test_LUQ_8, TestRestoreCorrectlyLargeMaxGroupedParallel) {
    float arr[81];
    int8_t q_arr[81] = {
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 15, 127, 127, 127, 127,
        15, 15, 15, 15, 15, 14, 14, 14,
        14, 14, 14, 13, 13, 13, 12, 11,
        127, 11, 12, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 15, 15, 15, 15,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 15,
        10, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15
    };
    int mask_count = 6;
    __mmask16 signs[6] = {
        65535, 65535, 288, 5, 1, 1
    };
    float scale[6] = {3.05176e+33, 0.000488281, 0.000457764, 3.05176e+33, 0.000976562, 3.05176e-05};
    float dq_const[6];
    float expected_dq_arr[81] = {
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, -1.0000e+38, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        -16, -16, -16, -16, -16, -8, -8, -8,
        -8, -8, -8, -4, -4, -4, -2, -1,
        0.0000, 0.9375, 1.8750, 3.7500, 3.7500, -3.7500, 7.5000, 7.5000,
        -7.5000, 7.5000, 7.5000, 7.5000, 15.0000, 15.0000, 15.0000, 15.0000,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+38,
        -1, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32,
        -1
    };
    union quantization::Quantization_Input<int8_t> input
        = {.luq_grouped_input={arr, q_arr, 81, scale, dq_const, mask_count, signs}};
    quantization::LUQ8Strategy strategy = quantization::LUQ8Strategy(4);

    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 81; i++) {
        ASSERT_THAT(input.luq_grouped_input.dq_values[i], testing::FloatNear(expected_dq_arr[i], fabs(expected_dq_arr[i] * 0.01f)));
    }
}