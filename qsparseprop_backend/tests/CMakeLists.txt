if (TESTING)
    # TODO: Remove test_gtest.cpp once you start writing real tests
    set(
            TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tester.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_simdmath.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_standard_quantization_strategy.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_sawb_quantization_strategy.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_dithered_quantization_strategy.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_luq_strategy.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_conv2d_stride_1.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_conv2d_stride_2.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_linear.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gtest.cpp
    )

    # Add subdirectories with code to use here
    # add_subdirectory(include) etc.

    add_executable(tester ${TEST_SOURCES})

    target_link_libraries(
            tester PUBLIC
            qsparse_core
            GTest::GTest
            GMock::GMock
    )
endif()