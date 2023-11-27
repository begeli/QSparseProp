#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/quantization/standard_quantization8_strategy.h"

/**
 * To the potential readers of this file (or future me)
 * I apologize for the disgusting test cases.
 * I am new at writing readable test cases. (This line reads like a tasteless joke.)
 * */

void getStandardQuantization8Data(
    float input_arr1[],
    float expected_arr1[],
    int8_t expected_q_arr1[],
    float& expected_scale
) {
    float arr1[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    float dq_arr_exp[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    int8_t q_arr_exp[33] = {
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93, 110, 127,
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93,110, 127,
        127
    };
    int size = 33;
    std::memcpy(input_arr1, arr1, sizeof(float) * size);
    std::memcpy(expected_arr1, dq_arr_exp, sizeof(float) * size);
    std::memcpy(expected_q_arr1, q_arr_exp, sizeof(int8_t) * size);
    expected_scale = 0.117647;
}

void getStandardQuantization8AllEqData(
        float input_arr1[],
        int8_t expected_q_arr1[]
) {
    float arr1[33] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f,
    };
    int8_t q_arr_exp[33] = {
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128
    };
    int size = 33;
    std::memcpy(input_arr1, arr1, sizeof(float) * size);
    std::memcpy(expected_q_arr1, q_arr_exp, sizeof(int8_t) * size);
}

TEST(Test_Standard_Quantization_8, TestQuantizeCorrectly) {
    float expected_scale;
    float arr1[33];
    float dq_arr_exp[33];
    int8_t q_arr1[33];
    int8_t q_arr_exp[33];
    getStandardQuantization8Data(arr1, dq_arr_exp, q_arr_exp, expected_scale);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, 0.0f, 0.0f}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize(input);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], q_arr_exp[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(expected_scale, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeCorrectlyParallel) {
    float expected_scale;
    float arr1[33];
    float dq_arr_exp[33];
    int8_t q_arr1[33];
    int8_t q_arr_exp[33];
    getStandardQuantization8Data(arr1, dq_arr_exp, q_arr_exp, expected_scale);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, 0.0f, 0.0f}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], q_arr_exp[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(expected_scale, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeCorrectlyGrouped) {
    float arr1[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    int8_t q_arr1[33];
    int8_t q_arr_exp[33] = {
        -128, -103, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -103, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.117647f, 0.117647f, 1.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    float expected_dq_const[3] = {1.44118f, 1.44118f, -141.5f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped(input, 16, 4);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], q_arr_exp[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dq_const[i], 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeCorrectlyGroupedParallel) {
    float arr1[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    int8_t q_arr1[33];
    int8_t q_arr_exp[33] = {
        -128, -103, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -103, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128
    };
    float scale[3] = {0.0f, 0.0f, 0.0f};
    float expected_scale[3] = {0.117647f, 0.117647f, 1.0f};
    float dq_const[3] = {0.0f, 0.0f, 0.0f};
    float expected_dq_const[3] = {1.44118f, 1.44118f, -141.5f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped_parallel(input, 16, 4);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], q_arr_exp[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dq_const[i], 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectly) {
    float scale = 0.117647;
    float dq_const = 1.44118;
    float expected_scale;
    float arr1[33];
    float dq_arr_exp[33];
    int8_t q_arr1[33];
    getStandardQuantization8Data(arr1, dq_arr_exp, q_arr1, expected_scale);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(dq_arr_exp[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyParallel) {
    float scale = 0.117647;
    float dq_const = 1.44118;
    float expected_scale;
    float arr1[33];
    float dq_arr_exp[33];
    int8_t q_arr1[33];
    getStandardQuantization8Data(arr1, dq_arr_exp, q_arr1, expected_scale);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_parallel(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(dq_arr_exp[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyGrouped) {
    float scale[3] = {0.117647f, 0.117647f, 1.0f};
    float dq_const[3] = {1.44118f, 1.44118f, -141.5f};
    float arr1[33];
    float dq_arr_exp[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    int8_t q_arr1[33] = {
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93, 110, 127,
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93,110, 127,
        -128
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(dq_arr_exp[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyGroupedParallel) {
    float scale[3] = {0.117647f, 0.117647f, 1.0f};
    float dq_const[3] = {1.44118f, 1.44118f, -141.5f};
    float arr1[33];
    float dq_arr_exp[33] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f,
        13.5f
    };
    int8_t q_arr1[33] = {
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93, 110, 127,
        -128, -102, -94, -77,
        -60, -43, -26, -9,
        8, 25, 42, 59,
        76, 93,110, 127,
        -128
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(dq_arr_exp[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyVectorized) {
    int8_t q_arr_exp[64] = {
        -128, -102, -94, -77,-60,-43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59,76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127
    };
    float dq_arr_expected[64] = {
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f
    };
    float dq_arr[64];
    float scale = 0.117647;
    float dq_const = 1.44118;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr_exp, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore(input);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(dq_arr_expected[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyVectorizedParallel) {
    int8_t q_arr_exp[64] = {
        -128, -102, -94, -77,-60,-43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59,76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127
    };
    float dq_arr_expected[64] = {
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f
    };
    float dq_arr[64];
    float scale = 0.117647;
    float dq_const = 1.44118;
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={dq_arr, q_arr_exp, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_parallel(input);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(dq_arr_expected[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyVectorizedGrouped) {
    int8_t q_arr_exp[64] = {
        -128, -102, -94, -77,-60,-43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59,76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127
    };
    float dq_arr_expected[64] = {
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f
    };
    float dq_arr[64];
    float scale[4] = {0.117647, 0.117647, 0.117647, 0.117647};
    float dq_const[4] = {1.44118, 1.44118, 1.44118, 1.44118};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr_exp, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(dq_arr_expected[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreCorrectlyVectorizedGroupedParallel) {
    int8_t q_arr_exp[64] = {
        -128, -102, -94, -77,-60,-43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59,76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127,
        -128, -102, -94, -77, -60, -43, -26, -9,
        8, 25, 42, 59, 76, 93, 110, 127
    };
    float dq_arr_expected[64] = {
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f,
        -16.5f, -13.5f, -12.5f, -10.5f, -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f
    };
    float dq_arr[64];
    float scale[4] = {0.117647, 0.117647, 0.117647, 0.117647};
    float dq_const[4] = {1.44118, 1.44118, 1.44118, 1.44118};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={dq_arr, q_arr_exp, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(dq_arr[i], testing::FloatNear(dq_arr_expected[i], 0.2f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeAllEq) {
    float arr1[33];
    int8_t expected_q_arr1[33];
    int8_t q_arr1[33];
    getStandardQuantization8AllEqData(arr1, expected_q_arr1);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, 0.0f, 0.0f}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize(input);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], expected_q_arr1[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(1.0f, 0.001f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(-133.0f, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeAllEqParallel) {
    float arr1[33];
    int8_t expected_q_arr1[33];
    int8_t q_arr1[33];
    getStandardQuantization8AllEqData(arr1, expected_q_arr1);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, 0.0f, 0.0f}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], expected_q_arr1[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(1.0f, 0.001f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(-133.0f, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeAllEqGrouped) {
    float arr1[33] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f,
    };
    int8_t expected_q_arr1[33] = {
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128
    };
    int8_t q_arr1[33];
    float scale[3];
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped(input, 16, 4);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], expected_q_arr1[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(1.0f, 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(-133.0f, 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeAllEqGroupedParallel) {
    float arr1[33] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f,
    };
    int8_t expected_q_arr1[33] = {
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128
    };
    int8_t q_arr1[33];
    float scale[3];
    float dq_const[3];
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped_parallel(input, 16, 4);

    for (int i = 0; i < 33; i++) {
        EXPECT_EQ(q_arr1[i], expected_q_arr1[i]);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(1.0f, 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(-133.0f, 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAllEq) {
    float scale = 1.0f;
    float dq_const = -133.0f;
    float arr1[33];
    int8_t q_arr1[33];
    getStandardQuantization8AllEqData(arr1, q_arr1);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(5.0f, 0.0001f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAllEqParallel) {
    float scale = 1.0f;
    float dq_const = -133.0f;
    float arr1[33];
    int8_t q_arr1[33];
    getStandardQuantization8AllEqData(arr1, q_arr1);
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_parallel(input);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(5.0f, 0.0001f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAllEqGrouped) {
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float dq_const[3] = {-133.0f, -133.0f, -133.0f};
    float arr1[33] = {
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        5.0f, 5.0f, 5.0f
    };
    int8_t q_arr1[33] = {
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128, -128, -128, -128,
        -128
    };
    getStandardQuantization8AllEqData(arr1, q_arr1);
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr1, q_arr1, 33, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 33; i++) {
        ASSERT_THAT(arr1[i], testing::FloatNear(5.0f, 0.0001f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeAsymmetric) {
    float arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    int8_t q_arr[64];
    int8_t expected_q_arr[64] = {
        -128, -124, -120, -116, -112, -108, -104, -100,
        -96, -92, -88, -83, -79, -75, -71, -67,
        -63, -59, -55, -51, -47, -43, -39, -35,
        -31, -27, -23, -19, -15, -11, -7, -3,
        2, 6, 10, 14, 18, 22, 26, 30,
        34, 38, 42, 46, 50, 54, 58, 62,
        66, 70, 74, 78, 82, 87, 91, 95,
        99, 103, 107, 111, 115, 119, 123, 127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 64, 0.0f, 0.0f}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize(input);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.247059f, 0.001f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(-32.6235f, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeAsymmetricParallel) {
    float arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    int8_t q_arr[64];
    int8_t expected_q_arr[64] = {
        -128, -124, -120, -116, -112, -108, -104, -100,
        -96, -92, -88, -83, -79, -75, -71, -67,
        -63, -59, -55, -51, -47, -43, -39, -35,
        -31, -27, -23, -19, -15, -11, -7, -3,
        2, 6, 10, 14, 18, 22, 26, 30,
        34, 38, 42, 46, 50, 54, 58, 62,
        66, 70, 74, 78, 82, 87, 91, 95,
        99, 103, 107, 111, 115, 119, 123, 127
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 64, 0.0f, 0.0f}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.247059f, 0.001f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(-32.6235f, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestQuantizeAsymmetricGrouped) {
    float arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    int8_t q_arr[64];
    int8_t expected_q_arr[64] = {
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
    };
    float scale[4];
    float expected_scale[4] = {0.0588235f, 0.0588235f, 0.0588235f, 0.0588235f};
    float dq_const[4];
    float expected_dq_const[4] = {-8.52941f, -24.5294f, -40.5294f, -56.5294f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 64, scale, dq_const}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped(input, 16, 4);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 4; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dq_const[i], 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeAsymmetricGroupedParallel) {
    float arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    int8_t q_arr[64];
    int8_t expected_q_arr[64] = {
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
    };
    float scale[4];
    float expected_scale[4] = {0.0588235f, 0.0588235f, 0.0588235f, 0.0588235f};
    float dq_const[4];
    float expected_dq_const[4] = {-8.52941f, -24.5294f, -40.5294f, -56.5294f};
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 64, scale, dq_const}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_grouped_parallel(input, 16, 4);

    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    for (int i = 0; i < 4; i++) {
        ASSERT_THAT(input.std_grouped_input.scale[i], testing::FloatNear(expected_scale[i], 0.001f));
        ASSERT_THAT(input.std_grouped_input.dequantization_const[i], testing::FloatNear(expected_dq_const[i], 0.001f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAsymmetric) {
    float scale = 0.247059;
    float dq_const = -32.6235;
    int8_t q_arr[64] = {
        -127, -123, -119, -115, -111, -107, -103, -99,
        -95, -91, -87, -83, -79, -75, -71, -67,
        -63, -59, -55, -51, -47, -42, -38, -34,
        -30, -26, -22, -18, -14, -10, -6, -2,
        1, 5, 9, 13, 17, 21, 25, 29,
        33, 37, 42, 46, 50, 54, 58, 62,
        66, 70, 74, 78, 82, 86, 90, 94,
        98, 102, 106, 110, 114, 118, 122, 127
    };
    float arr[64];
    float expected_dq_arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore(input);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.3f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAsymmetricParallel) {
    float scale = 0.247059;
    float dq_const = -32.6235;
    int8_t q_arr[64] = {
        -127, -123, -119, -115, -111, -107, -103, -99,
        -95, -91, -87, -83, -79, -75, -71, -67,
        -63, -59, -55, -51, -47, -42, -38, -34,
        -30, -26, -22, -18, -14, -10, -6, -2,
        1, 5, 9, 13, 17, 21, 25, 29,
        33, 37, 42, 46, 50, 54, 58, 62,
        66, 70, 74, 78, 82, 86, 90, 94,
        98, 102, 106, 110, 114, 118, 122, 127
    };
    float arr[64];
    float expected_dq_arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_parallel(input);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.3f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAsymmetricGrouped) {
    float scale[4] = {0.0588235f, 0.0588235f, 0.0588235f, 0.0588235f};
    float dq_const[4] = {-8.52941f, -24.5294f, -40.5294f, -56.5294f};
    int8_t q_arr[64] = {
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127
    };
    float arr[64];
    float expected_dq_arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped(input, 16, 4);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.3f));
    }
}

TEST(Test_Standard_Quantization_8, TestRestoreAsymmetricGroupedParallel) {
    float scale[4] = {0.0588235f, 0.0588235f, 0.0588235f, 0.0588235f};
    float dq_const[4] = {-8.52941f, -24.5294f, -40.5294f, -56.5294f};
    int8_t q_arr[64] = {
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127,
        -128, -111, -94, -77, -60, -43, -26,
        -9, 8, 25, 42, 59, 76, 93, 110, 127
    };
    float arr[64];
    float expected_dq_arr[64] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
        57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_grouped_input={arr, q_arr, 64, scale, dq_const}};

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_grouped_parallel(input, 16, 4);
    for (int i = 0; i < 64; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.3f));
    }
}

TEST(Test_Standard_Quantization_8, TestQuantizeMultipleIterationsParallel) {
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
        -82, -8, -50, 88, 127, -34, -7, -79,
        -110, -30, -126, 105, -51, -84, -48, -114,
        83, -25, -10, -30, 66, -69, -69, 56,
        -84, -70, -95, -105, 122, 9, -128, 103,
        115, 12, 7, -22, -3, 68, 11, 89,
        -127, -114, 61, -120, -116, 33, -110, -64,
        -90, -47, -33, -45, 52, -32, -27, 109,
        -26, 101, 55, -47, -115, -71, 91, 38,
        104, -113, 49, -70, -75, -58, 80, 86,
        -33, 39, 29, -61, 15, 21, -36, -87,
        74, 2, 12, -110, 119, 87, -91, 75,
        56, -15, -69, 101, -117, -33, -114, 35,
        15, -36, 90, -30, -7, 71, 96, 15,
        90, 49, 94, 2, 108, -24, -40, 27,
        116, 47, -118, -91, 105, 12, -109, -82,
        92, 56, -23, 22, -49, 3, 95, 101,
        -34, 35, -41, 105, -13, 6, 10, 8,
        8, 122, 40, -48
    };
    union quantization::Quantization_Input<int8_t> input
        = {.std_quantization_input={arr, q_arr, 140, 0.0f, 0.0f}};
    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.quantize_parallel(input);

    for (int i = 0; i < 140; i++) {
        EXPECT_EQ(q_arr[i], expected_q_arr[i]);
    }
    ASSERT_THAT(input.std_quantization_input.scale, testing::FloatNear(0.00775686f, 0.001f));
    ASSERT_THAT(input.std_quantization_input.dequantization_const, testing::FloatNear(-0.0055784f, 0.001f));
}

TEST(Test_Standard_Quantization_8, TestRestoreMultipleIterationsParallel) {
    float scale = 0.00775686f;
    float dq_const = -0.0055784f;
    int8_t q_arr[140] = {
        -82, -8, -50, 88, 127, -34, -7, -79,
        -110, -30, -126, 105, -51, -84, -48, -114,
        83, -25, -10, -30, 66, -69, -69, 56,
        -84, -70, -95, -105, 122, 9, -128, 103,
        115, 12, 7, -22, -3, 68, 11, 89,
        -127, -114, 61, -120, -116, 33, -110, -64,
        -90, -47, -33, -45, 52, -32, -27, 109,
        -26, 101, 55, -47, -115, -71, 91, 38,
        104, -113, 49, -70, -75, -58, 80, 86,
        -33, 39, 29, -61, 15, 21, -36, -87,
        74, 2, 12, -110, 119, 87, -91, 75,
        56, -15, -69, 101, -117, -33, -114, 35,
        15, -36, 90, -30, -7, 71, 96, 15,
        90, 49, 94, 2, 108, -24, -40, 27,
        116, 47, -118, -91, 105, 12, -109, -82,
        92, 56, -23, 22, -49, 3, 95, 101,
        -34, 35, -41, 105, -13, 6, 10, 8,
        8, 122, 40, -48
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

    quantization::StandardQuantization8Strategy strategy
        = quantization::StandardQuantization8Strategy(-128.0f, 127.0f);
    strategy.restore_parallel(input);
    for (int i = 0; i < 140; i++) {
        ASSERT_THAT(arr[i], testing::FloatNear(expected_dq_arr[i], 0.01f));
    }
}