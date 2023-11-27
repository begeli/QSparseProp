#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/utils/tensor_utils.h"

TEST(Test_Tensor_Utils, TestTransposeCorrectly) {
    int block_size = 16;
    int M = 64;
    int N = 48;
    float arr[M * N];
    float arr_t[N * M];
    float expected_arr_t[N * M];
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            arr[row * N + col] = row * M + col;
            expected_arr_t[col * M + row] = row * M + col;
            arr_t[col * M + row] = 0.0f;
        }
    }

    transpose(arr, arr_t, M, N, block_size);
    for (int i = 0; i < N * M; i++) {
        ASSERT_THAT(arr_t[i], testing::FloatNear(expected_arr_t[i], 0.000001f));
    }
}