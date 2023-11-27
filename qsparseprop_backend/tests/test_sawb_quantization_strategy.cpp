#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/quantization/sawb_quantization8_strategy.h"

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyPos) {
    float arr[65] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        64.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        2, 5, 7, 9, 11, 14, 16, 18,
        20, 23, 25, 27, 29, 32, 34, 36,
        38, 41, 43, 45, 47, 50, 52, 54,
        56, 59, 61, 63, 65, 68, 70, 72,
        74, 77, 79, 81, 83, 86, 88, 90,
        92, 95, 97, 99, 101, 104, 106, 108,
        110, 113, 115, 117, 119, 122, 124, 126,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.4439f, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyPosParallel) {
    float arr[65] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        64.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        2, 5, 7, 9, 11, 14, 16, 18,
        20, 23, 25, 27, 29, 32, 34, 36,
        38, 41, 43, 45, 47, 50, 52, 54,
        56, 59, 61, 63, 65, 68, 70, 72,
        74, 77, 79, 81, 83, 86, 88, 90,
        92, 95, 97, 99, 101, 104, 106, 108,
        110, 113, 115, 117, 119, 122, 124, 126,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.4439f, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyPosGrouped) {
    float arr[65] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        64.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        5, 9, 14, 19, 23, 28, 32, 37,
        42, 46, 51, 56, 60, 65, 70, 74,
        79, 84, 88, 93, 97, 102, 107, 111,
        116, 121, 125, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.215552f, 0.0446248f, -0.0501958};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dq_const}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyPosGroupedParallel) {
    float arr[65] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        64.0
    };
    int8_t q_arr[65];
    int8_t expected_q_arr[65] = {
        5, 9, 14, 19, 23, 28, 32, 37,
        42, 46, 51, 56, 60, 65, 70, 74,
        79, 84, 88, 93, 97, 102, 107, 111,
        116, 121, 125, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.215552f, 0.0446248f, -0.0501958};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dq_const}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyPos) {
    int8_t q_arr[65] = {
        2, 5, 7, 9, 11, 14, 16, 18,
        20, 23, 25, 27, 29, 32, 34, 36,
        38, 41, 43, 45, 47, 50, 52, 54,
        56, 59, 61, 63, 65, 68, 70, 72,
        74, 77, 79, 81, 83, 86, 88, 90,
        92, 95, 97, 99, 101, 104, 106, 108,
        110, 113, 115, 117, 119, 122, 124, 126,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        0.8877,  2.2194,  3.1071,  3.9948,  4.8826,  6.2142,  7.1019,  7.9897,
        8.8774, 10.2090, 11.0968, 11.9845, 12.8722, 14.2038, 15.0916, 15.9793,
        16.8671, 18.1987, 19.0864, 19.9742, 20.8619, 22.1935, 23.0812, 23.9690,
        24.8567, 26.1883, 27.0761, 27.9638, 28.8516, 30.1832, 31.0709, 31.9587,
        32.8464, 34.1780, 35.0657, 35.9535, 36.8412, 38.1728, 39.0606, 39.9483,
        40.8361, 42.1677, 43.0554, 43.9431, 44.8309, 46.1625, 47.0502, 47.9380,
        48.8257, 50.1573, 51.0451, 51.9328, 52.8205, 54.1522, 55.0399, 55.9276,
        56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715,
        56.3715
    };
    float scale = 0.4439f;
    float dequantization_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyPosParallel) {
    int8_t q_arr[65] = {
        2, 5, 7, 9, 11, 14, 16, 18,
        20, 23, 25, 27, 29, 32, 34, 36,
        38, 41, 43, 45, 47, 50, 52, 54,
        56, 59, 61, 63, 65, 68, 70, 72,
        74, 77, 79, 81, 83, 86, 88, 90,
        92, 95, 97, 99, 101, 104, 106, 108,
        110, 113, 115, 117, 119, 122, 124, 126,
        127, 127, 127, 127, 127, 127, 127, 127,
        127
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        0.8877,  2.2194,  3.1071,  3.9948,  4.8826,  6.2142,  7.1019,  7.9897,
        8.8774, 10.2090, 11.0968, 11.9845, 12.8722, 14.2038, 15.0916, 15.9793,
        16.8671, 18.1987, 19.0864, 19.9742, 20.8619, 22.1935, 23.0812, 23.9690,
        24.8567, 26.1883, 27.0761, 27.9638, 28.8516, 30.1832, 31.0709, 31.9587,
        32.8464, 34.1780, 35.0657, 35.9535, 36.8412, 38.1728, 39.0606, 39.9483,
        40.8361, 42.1677, 43.0554, 43.9431, 44.8309, 46.1625, 47.0502, 47.9380,
        48.8257, 50.1573, 51.0451, 51.9328, 52.8205, 54.1522, 55.0399, 55.9276,
        56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715, 56.3715,
        56.3715
    };
    float scale = 0.4439f;
    float dequantization_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyPosGrouped) {
    int8_t q_arr[65] = {
        5, 9, 14, 19, 23, 28, 32, 37,
        42, 46, 51, 56, 60, 65, 70, 74,
        79, 84, 88, 93, 97, 102, 107, 111,
        116, 121, 125, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        -128
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        1.07776, 1.93997, 3.01773, 4.09549, 4.9577, 6.03546, 6.89766, 7.97542,
        9.05318, 9.91539, 10.9932, 12.0709, 12.9331, 14.0109, 15.0886, 15.9508,
        17.0286, 18.1064, 18.9686, 20.0463, 20.9085, 21.9863, 23.0641, 23.9263,
        25.004, 26.0818, 26.944, 27.3751, 27.3751, 27.3751, 27.3751, 27.3751,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 6.4251
    };
    float scale[3] = {0.215552f, 0.0446248f, -0.0502};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyPosGroupedParallel) {
    int8_t q_arr[65] = {
        5, 9, 14, 19, 23, 28, 32, 37,
        42, 46, 51, 56, 60, 65, 70, 74,
        79, 84, 88, 93, 97, 102, 107, 111,
        116, 121, 125, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127,
        -128
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        1.07776, 1.93997, 3.01773, 4.09549, 4.9577, 6.03546, 6.89766, 7.97542,
        9.05318, 9.91539, 10.9932, 12.0709, 12.9331, 14.0109, 15.0886, 15.9508,
        17.0286, 18.1064, 18.9686, 20.0463, 20.9085, 21.9863, 23.0641, 23.9263,
        25.004, 26.0818, 26.944, 27.3751, 27.3751, 27.3751, 27.3751, 27.3751,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735, 5.66735,
        5.66735, 6.4251
    };
    float scale[3] = {0.215552f, 0.0446248f, -0.0502};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyMixed) {
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
        -128, -128, -128, -128, -124, -119, -115, -111,
        -106, -102, -97, -93, -89, -84, -80, -75,
        -71, -66, -62, -58, -53, -49, -44, -40,
        -35, -31, -27, -22, -18, -13, -9, -4,
        0, 4, 9, 13, 18, 22, 27, 31,
        35, 40, 44, 49, 53, 58, 62, 66,
        71, 75, 80, 84, 89, 93, 97, 102,
        106, 111, 115, 119, 124, 127, 127, 127,
        127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.2260, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyMixedParallel) {
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
        -128, -128, -128, -128, -124, -119, -115, -111,
        -106, -102, -97, -93, -89, -84, -80, -75,
        -71, -66, -62, -58, -53, -49, -44, -40,
        -35, -31, -27, -22, -18, -13, -9, -4,
        0, 4, 9, 13, 18, 22, 27, 31,
        35, 40, 44, 49, 53, 58, 62, 66,
        71, 75, 80, 84, 89, 93, 97, 102,
        106, 111, 115, 119, 124, 127, 127, 127,
        127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 65, 0.0f, 0.0f}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.2260, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyMixedGrouped) {
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
        -128, -128, -128, -128, -128, -125, -121, -116,
        -111, -107, -102, -97, -93, -88, -84, -79,
        -74, -70, -65, -60, -56, -51, -46, -42,
        -37, -32, -28, -23, -19, -14, -9, -5,
        0, 4, 9, 13, 17, 22, 26, 31,
        35, 39, 44, 48, 52, 57, 61, 65,
        70, 74, 79, 83, 87, 92, 96, 100, 105,
        109, 114, 118, 122, 127, 127, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.215552f, 0.229048f, -0.0250979};
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dq_const}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyMixedGroupedParallel) {
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
        -128, -128, -128, -128, -128, -125, -121, -116,
        -111, -107, -102, -97, -93, -88, -84, -79,
        -74, -70, -65, -60, -56, -51, -46, -42,
        -37, -32, -28, -23, -19, -14, -9, -5,
        0, 4, 9, 13, 17, 22, 26, 31,
        35, 39, 44, 48, 52, 57, 61, 65,
        70, 74, 79, 83, 87, 92, 96, 100, 105,
        109, 114, 118, 122, 127, 127, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.215552f, 0.229048f, -0.0250979};
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale, dq_const}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyMixed) {
    int8_t q_arr[65] = {
        -128, -128, -128, -128, -124, -119, -115, -111,
        -106, -102, -97, -93, -89, -84, -80, -75,
        -71, -66, -62, -58, -53, -49, -44, -40,
        -35, -31, -27, -22, -18, -13, -9, -4,
        0, 4, 9, 13, 18, 22, 27, 31,
        35, 40, 44, 49, 53, 58, 62, 66,
        71, 75, 80, 84, 89, 93, 97, 102,
        106, 111, 115, 119, 124, 127, 127, 127,
        127
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -28.9260, -28.9260, -28.9260, -28.9260, -28.0221, -26.8922, -25.9882,
        -25.0843, -23.9544, -23.0504, -21.9205, -21.0166, -20.1126, -18.9827,
        -18.0788, -16.9489, -16.0449, -14.9150, -14.0111, -13.1071, -11.9772,
        -11.0733,  -9.9433,  -9.0394,  -7.9095,  -7.0055,  -6.1016,  -4.9717,
        -4.0677,  -2.9378,  -2.0339,  -0.9039,   0.0000,   0.9039,   2.0339,
        2.9378,   4.0677,   4.9717,   6.1016,   7.0055,   7.9095,   9.0394,
        9.9433,  11.0733,  11.9772,  13.1071,  14.0111,  14.9150,  16.0449,
        16.9489,  18.0788,  18.9827,  20.1126,  21.0166,  21.9205,  23.0504,
        23.9544,  25.0843,  25.9882,  26.8922,  28.0221,  28.7001,  28.7001,
        28.7001,  28.7001
    };
    float scale = 0.2260;
    float dequantization_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyMixedParallel) {
    int8_t q_arr[65] = {
        -128, -128, -128, -128, -124, -119, -115, -111,
        -106, -102, -97, -93, -89, -84, -80, -75,
        -71, -66, -62, -58, -53, -49, -44, -40,
        -35, -31, -27, -22, -18, -13, -9, -4,
        0, 4, 9, 13, 18, 22, 27, 31,
        35, 40, 44, 49, 53, 58, 62, 66,
        71, 75, 80, 84, 89, 93, 97, 102,
        106, 111, 115, 119, 124, 127, 127, 127,
        127
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -28.9260, -28.9260, -28.9260, -28.9260, -28.0221, -26.8922, -25.9882,
        -25.0843, -23.9544, -23.0504, -21.9205, -21.0166, -20.1126, -18.9827,
        -18.0788, -16.9489, -16.0449, -14.9150, -14.0111, -13.1071, -11.9772,
        -11.0733,  -9.9433,  -9.0394,  -7.9095,  -7.0055,  -6.1016,  -4.9717,
        -4.0677,  -2.9378,  -2.0339,  -0.9039,   0.0000,   0.9039,   2.0339,
        2.9378,   4.0677,   4.9717,   6.1016,   7.0055,   7.9095,   9.0394,
        9.9433,  11.0733,  11.9772,  13.1071,  14.0111,  14.9150,  16.0449,
        16.9489,  18.0788,  18.9827,  20.1126,  21.0166,  21.9205,  23.0504,
        23.9544,  25.0843,  25.9882,  26.8922,  28.0221,  28.7001,  28.7001,
        28.7001,  28.7001
    };
    float scale = 0.2260;
    float dequantization_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyMixedGrouped) {
    int8_t q_arr[65] = {
        -128, -128, -128, -128, -128, -125, -121, -116,
        -111, -107, -102, -97, -93, -88, -84, -79,
        -74, -70, -65, -60, -56, -51, -46, -42,
        -37, -32, -28, -23, -19, -14, -9, -5,
        0, 4, 9, 13, 17, 22, 26, 31,
        35, 39, 44, 48, 52, 57, 61, 65,
        70, 74, 79, 83, 87, 92, 96, 100, 105,
        109, 114, 118, 122, 127, 127, 127,
        -128
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -27.5907, -27.5907, -27.5907, -27.5907, -27.5907, -26.944, -26.0818, -25.004,
        -23.9263, -23.0641, -21.9863, -20.9085, -20.0463, -18.9686, -18.1064, -17.0286,
        -15.9508, -15.0886, -14.0109, -12.9331, -12.0709, -10.9932, -9.91539, -9.05318,
        -7.97542, -6.89766, -6.03546, -4.9577, -4.09549, -3.01773, -1.93997, -1.07776,
        0, 0.916192, 2.06143, 2.97762, 3.89382, 5.03906, 5.95525, 7.10049, 8.01668,
        8.93287, 10.0781, 10.9943, 11.9105, 13.0557, 13.9719, 14.8881,
        16.0334, 16.9496, 18.0948, 19.011, 19.9272, 21.0724, 21.9886, 22.9048,
        24.05, 24.9662, 26.1115, 27.0277, 27.9439, 29.0891, 29.0891, 29.0891,
        3.21253
    };
    float scale[3] = {0.215552f, 0.229048f, -0.0250979};
    float dequantization_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyMixedGroupedParallel) {
    int8_t q_arr[65] = {
        -128, -128, -128, -128, -128, -125, -121, -116,
        -111, -107, -102, -97, -93, -88, -84, -79,
        -74, -70, -65, -60, -56, -51, -46, -42,
        -37, -32, -28, -23, -19, -14, -9, -5,
        0, 4, 9, 13, 17, 22, 26, 31,
        35, 39, 44, 48, 52, 57, 61, 65,
        70, 74, 79, 83, 87, 92, 96, 100, 105,
        109, 114, 118, 122, 127, 127, 127,
        -128
    };
    float dq_arr[65];
    float expected_dq_arr[65] = {
        -27.5907, -27.5907, -27.5907, -27.5907, -27.5907, -26.944, -26.0818, -25.004,
        -23.9263, -23.0641, -21.9863, -20.9085, -20.0463, -18.9686, -18.1064, -17.0286,
        -15.9508, -15.0886, -14.0109, -12.9331, -12.0709, -10.9932, -9.91539, -9.05318,
        -7.97542, -6.89766, -6.03546, -4.9577, -4.09549, -3.01773, -1.93997, -1.07776,
        0, 0.916192, 2.06143, 2.97762, 3.89382, 5.03906, 5.95525, 7.10049, 8.01668,
        8.93287, 10.0781, 10.9943, 11.9105, 13.0557, 13.9719, 14.8881,
        16.0334, 16.9496, 18.0948, 19.011, 19.9272, 21.0724, 21.9886, 22.9048,
        24.05, 24.9662, 26.1115, 27.0277, 27.9439, 29.0891, 29.0891, 29.0891,
        3.21253
    };
    float scale[3] = {0.215552f, 0.229048f, -0.0250979};
    float dequantization_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dequantization_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(expected_dq_arr[i], 0.1f));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyAllZero) {
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
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(2.0f / 255.0f, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyAllZeroParallel) {
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
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(2.0f / 255.0f, 0.01));
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyAllZeroGrouped) {
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
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale,dq_const}};
    quantization::SAWBQuantization8Strategy strategy
            = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(2.0f / 255.0f, 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeCorrectlyAllZeroGroupedParallel) {
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
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 65, scale,dq_const}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.quantize_grouped_parallel(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        EXPECT_EQ(q_arr[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(2.0f / 255.0f, 0.01));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyAllZero) {
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
    float scale = 2.0f / 255.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.0000001f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyAllZeroParallel) {
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
    float scale = 2.0f / 255.0f;
    float dq_const = 0.0f;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_parallel(input);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.0000001f));
    }
}

TEST(Test_SAWB_Quantization_8, TestRestoreCorrectlyAllZeroGrouped) {
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
    float scale[3] = {2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f};
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr, 65, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1f, 12.2f);
    strategy.restore_grouped(input, 32, 5);

    for (int i = 0; i < 65; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(0.0f, 0.0000001f));
    }
}

TEST(Test_SAWB_Quantization_8, TestQuantizeMultipleIterationsParallel) {
    float arr[140] = {
        -0.6329, -0.0565, -0.3814,  0.6864,  0.9907, -0.2580, -0.0510, -0.6081,
        -0.8515, -0.2309, -0.9719,  0.8228, -0.3931, -0.6471, -0.3635, -0.8780,
        0.6516, -0.1850, -0.0688, -0.2254,  0.5167, -0.5324, -0.5319,  0.4394,
        -0.6473, -0.5348, -0.7310, -0.8082,  0.9533,  0.0736, -0.9873,  0.8021,
        0.8995,  0.0970,  0.0592, -0.1652, -0.0159,  0.5322,  0.0939,  0.6943,
        -0.9772, -0.8824,  0.4821, -0.9290, -0.8929,  0.2647, -0.8502, -0.4876,
        -0.6957, -0.3607, -0.2518, -0.3463,  0.4063, -0.2426, -0.2017,  0.8519,
        -0.1979,  0.7925,  0.4334, -0.3562, -0.8854, -0.5455,  0.7124,  0.3000,
        0.8129, -0.8694,  0.3823, -0.5395, -0.5790, -0.4409,  0.6279,  0.6765,
        -0.2472,  0.3059,  0.2342, -0.4643,  0.1185,  0.1678, -0.2762, -0.6701,
        0.5791,  0.0203,  0.1014, -0.8467,  0.9264,  0.6781, -0.6984,  0.5881,
        0.4421, -0.1100, -0.5330,  0.7881, -0.9025, -0.2523, -0.8762,  0.2777,
        0.1257, -0.2737,  0.7039, -0.2254, -0.0484,  0.5583,  0.7529,  0.1202,
        0.7075,  0.3876,  0.7377,  0.0187,  0.8405, -0.1816, -0.3078,  0.2138,
        0.9082,  0.3673, -0.9125, -0.7028,  0.8229,  0.0952, -0.8399, -0.6306,
        0.7230,  0.4425, -0.1718,  0.1799, -0.3759,  0.0323,  0.7418,  0.7922,
        -0.2513,  0.2823, -0.3097,  0.8272, -0.0896,  0.0565,  0.0875,  0.0688,
        0.0715,  0.9551,  0.3231, -0.3621
    };
    int8_t q_arr[140];
    int8_t expected_q_arr[140] = {
        -86, -8, -52, 93, 127, -35, -7, -82,
        -115, -31, -128, 111, -53, -88, -49, -119,
        88, -25, -9, -31, 70, -72, -72, 60,
        -88, -72, -99, -110, 127, 10, -128, 109,
        122, 13, 8, -22, -2, 72, 13, 94,
        -128, -120, 65, -126, -121, 36, -115, -66,
        -94, -49, -34, -47, 55, -33, -27, 115,
        -27, 107, 59, -48, -120, -74, 97, 41,
        110, -118, 52, -73, -78, -60, 85, 92,
        -33, 41, 32, -63, 16, 23, -37, -91,
        78, 3, 14, -115, 126, 92, -95, 80,
        60, -15, -72, 107, -122, -34, -119, 38,
        17, -37, 95, -31, -7, 76, 102, 16,
        96, 53, 100, 3, 114, -25, -42, 29,
        123, 50, -124, -95, 112, 13, -114, -85,
        98, 60, -23, 24, -51, 4, 101, 107,
        -34, 38, -42, 112, -12, 8, 12, 9,
        10, 127, 44, -49
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 140, 0.0f, 0.0f}};
    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1, 12.2);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 140; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.00737969f, 0.001f));
}

TEST(Test_SAWB_Quantization_8, TestRestoreMultipleIterationsParallel) {
    float scale = 0.00737969f;
    float dq_const = 0.0f;
    int8_t q_arr[140] = {
        -86, -8, -52, 93, 127, -35, -7, -82,
        -115, -31, -128, 111, -53, -88, -49, -119,
        88, -25, -9, -31, 70, -72, -72, 60,
        -88, -72, -99, -110, 127, 10, -128, 109,
        122, 13, 8, -22, -2, 72, 13, 94,
        -128, -120, 65, -126, -121, 36, -115, -66,
        -94, -49, -34, -47, 55, -33, -27, 115,
        -27, 107, 59, -48, -120, -74, 97, 41,
        110, -118, 52, -73, -78, -60, 85, 92,
        -33, 41, 32, -63, 16, 23, -37, -91,
        78, 3, 14, -115, 126, 92, -95, 80,
        60, -15, -72, 107, -122, -34, -119, 38,
        17, -37, 95, -31, -7, 76, 102, 16,
        96, 53, 100, 3, 114, -25, -42, 29,
        123, 50, -124, -95, 112, 13, -114, -85,
        98, 60, -23, 24, -51, 4, 101, 107,
        -34, 38, -42, 112, -12, 8, 12, 9,
        10, 127, 44, -49
    };
    float arr[140];
    float expected_dq_arr[140] = {
        -0.6329, -0.0565, -0.3814,  0.6864,  0.9907, -0.2580, -0.0510, -0.6081,
        -0.8515, -0.2309, -0.9719,  0.8228, -0.3931, -0.6471, -0.3635, -0.8780,
        0.6516, -0.1850, -0.0688, -0.2254,  0.5167, -0.5324, -0.5319,  0.4394,
        -0.6473, -0.5348, -0.7310, -0.8082,  0.9533,  0.0736, -0.9873,  0.8021,
        0.8995,  0.0970,  0.0592, -0.1652, -0.0159,  0.5322,  0.0939,  0.6943,
        -0.9772, -0.8824,  0.4821, -0.9290, -0.8929,  0.2647, -0.8502, -0.4876,
        -0.6957, -0.3607, -0.2518, -0.3463,  0.4063, -0.2426, -0.2017,  0.8519,
        -0.1979,  0.7925,  0.4334, -0.3562, -0.8854, -0.5455,  0.7124,  0.3000,
        0.8129, -0.8694,  0.3823, -0.5395, -0.5790, -0.4409,  0.6279,  0.6765,
        -0.2472,  0.3059,  0.2342, -0.4643,  0.1185,  0.1678, -0.2762, -0.6701,
        0.5791,  0.0203,  0.1014, -0.8467,  0.9264,  0.6781, -0.6984,  0.5881,
        0.4421, -0.1100, -0.5330,  0.7881, -0.9025, -0.2523, -0.8762,  0.2777,
        0.1257, -0.2737,  0.7039, -0.2254, -0.0484,  0.5583,  0.7529,  0.1202,
        0.7075,  0.3876,  0.7377,  0.0187,  0.8405, -0.1816, -0.3078,  0.2138,
        0.9082,  0.3673, -0.9125, -0.7028,  0.8229,  0.0952, -0.8399, -0.6306,
        0.7230,  0.4425, -0.1718,  0.1799, -0.3759,  0.0323,  0.7418,  0.7922,
        -0.2513,  0.2823, -0.3097,  0.8272, -0.0896,  0.0565,  0.0875,  0.0688,
        0.0715,  0.9551,  0.3231, -0.3621
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 140, scale, dq_const}};

    quantization::SAWBQuantization8Strategy strategy
        = quantization::SAWBQuantization8Strategy(-128.0f, 127.0f, 12.1, 12.2);
    strategy.restore_parallel(input);
    for (int i = 0; i < 140; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.06f));
    }
}