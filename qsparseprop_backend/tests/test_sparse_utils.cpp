#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "src/utils/sparse_utils.h"

TEST(Test_Sparse, TestSparsifyConv2dWeightsCorrectly1) {
    int OC = 4;
    int IC = 4;
    int K = 3;
    float W[144] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f,
        4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 11.0f, 0.0f,
        0.0f, 0.0f, 12.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 14.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 15.0f, 0.0f, 0.0f, 0.0f,
        16.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    int W_idx_OC[5];
    int expected_W_idx_OC[5] = {
        0, 4, 8, 12, 16
    };
    int16_t W_idx_IC[20];
    int16_t expected_W_idx_IC[20] = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
    };
    uint8_t W_idx_X[16];
    uint8_t expected_W_idx_X[16] = {
        2, 1, 2, 0, 0, 1, 1, 0,
        2, 1, 2, 0, 1, 0, 1, 0
    };
    uint8_t W_idx_Y[16];
    uint8_t expected_W_idx_Y[16] = {
        0, 0, 2, 0, 2, 0, 2, 1,
        0, 1, 1, 2, 0, 1, 2, 0
    };
    float W_val[16];
    float expected_W_val[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };

    sparsify_conv2d(IC, OC, K, W, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(W_idx_OC[i], expected_W_idx_OC[i]);
    }

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(W_idx_IC[i], expected_W_idx_IC[i]);
    }

    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(W_idx_X[i], expected_W_idx_X[i]);
        EXPECT_EQ(W_idx_Y[i], expected_W_idx_Y[i]);
        ASSERT_THAT(W_val[i], testing::FloatNear(expected_W_val[i], 0.0001f));
    }
}

TEST(Test_Sparse, TestSparsifyConv2dWeightsCorrectly2) {
    int OC = 3;
    int IC = 2;
    int K = 3;
    float W[54] = {
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    int W_idx_OC[4];
    int expected_W_idx_OC[4] = {
        0, 1, 3, 6
    };
    int16_t W_idx_IC[9];
    int16_t expected_W_idx_IC[9] = {
        0, 1, 1, 0, 0, 2, 0, 2, 3
    };
    uint8_t W_idx_X[6];
    uint8_t expected_W_idx_X[6] = {
        0, 0, 1, 1, 1, 0
    };
    uint8_t W_idx_Y[6];
    uint8_t expected_W_idx_Y[6] = {
        1, 1, 1, 0, 2, 2
    };
    float W_val[6];
    float expected_W_val[6] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    };

    sparsify_conv2d(IC, OC, K, W, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val);

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(W_idx_OC[i], expected_W_idx_OC[i]);
    }

    for (int i = 0; i < 9; i++) {
        EXPECT_EQ(W_idx_IC[i], expected_W_idx_IC[i]);
    }

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(W_idx_X[i], expected_W_idx_X[i]);
        EXPECT_EQ(W_idx_Y[i], expected_W_idx_Y[i]);
        ASSERT_THAT(W_val[i], testing::FloatNear(expected_W_val[i], 0.0001f));
    }
}

TEST(Test_Sparse, TestDensifyConv2dWeightsCorrectly1) {
    int OC = 4;
    int IC = 4;
    int K = 3;
    float W[144];
    for (int i = 0; i < 144; i++) {
        W[i] = 0.0f;
    }
    float expected_W[144] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f,
        4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 11.0f, 0.0f,
        0.0f, 0.0f, 12.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 14.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 15.0f, 0.0f, 0.0f, 0.0f,
        16.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    int W_idx_OC[5] = {
        0, 4, 8, 12, 16
    };
    int16_t W_idx_IC[20] = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
    };
    uint8_t W_idx_X[16] = {
        2, 1, 2, 0, 0, 1, 1, 0,
        2, 1, 2, 0, 1, 0, 1, 0
    };
    uint8_t W_idx_Y[16] = {
        0, 0, 2, 0, 2, 0, 2, 1,
        0, 1, 1, 2, 0, 1, 2, 0
    };
    float W_val[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };

    densify_conv2d(IC, OC, K, W, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val);

    for (int i = 0; i < 144; i++) {
        EXPECT_EQ(W[i], expected_W[i]);
    }
}

TEST(Test_Sparse, TestDensifyConv2dWeightsCorrectly2) {
    int OC = 3;
    int IC = 2;
    int K = 3;
    float W[54];
    for (int i = 0; i < 54; i++) {
        W[i] = 0.0f;
    }
    float expected_W[54] = {
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    int W_idx_OC[4] = {
        0, 1, 3, 6
    };
    int16_t W_idx_IC[9] = {
        0, 1, 1, 0, 0, 2, 0, 2, 3
    };
    uint8_t W_idx_X[6] = {
        0, 0, 1, 1, 1, 0
    };
    uint8_t W_idx_Y[6] = {
        1, 1, 1, 0, 2, 2
    };
    float W_val[6] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    };

    densify_conv2d(IC, OC, K, W, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val);

    for (int i = 0; i < 54; i++) {
        EXPECT_EQ(W[i], expected_W[i]);
    }
}
