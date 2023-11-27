#include <pybind11/pybind11.h>
#include "qsparseprop_conv2d_wrapper.h"
#include "qsparseprop_linear_wrapper.h"
#include "qsparseprop_ditheredq_wrapper.h"
#include "qsparseprop_sawbq_wrapper.h"
#include "qsparseprop_stdq_wrapper.h"
#include "tensor_utils_wrapper.h"

#include "sparseprop_linear_wrapper.h"
#include "sparseprop_conv2d_wrapper.h"
#include "sparseprop_conv2d_over_on_wrapper.h"
#include "qsparseprop_conv2d_over_on_wrapper.h"

PYBIND11_MODULE(qsparseprop_backend, m) {
    m.doc() = "This python module contains AVX implementations of "
              "quantized sparse backward and forward propagation algorithms "
              "for linear and convolutional layers.";

    // Linear
    m.def(
        "sparse_linear_vectorized_forward",
        &sparse::sparse_linear_vectorized_forward_wrapper,
        "AVX512 implementation of sparse forward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_parallel_forward",
        &sparse::sparse_linear_vectorized_parallel_forward_wrapper,
        "AVX512 implementation of parallel sparse forward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_backward",
        &sparse::sparse_linear_vectorized_backward_wrapper,
        "AVX512 implementation of sparse backward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_parallel_backward",
        &sparse::sparse_linear_vectorized_parallel_backward_wrapper,
        "AVX512 implementation of parallel sparse backward propagation algorithm for linear layers."
    );

    m.def(
        "sparse_linear_vectorized_grouped_forward",
        &sparse::sparse_linear_vectorized_grouped_forward_wrapper,
        "AVX512 implementation of grouped sparse forward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_grouped_parallel_forward",
        &sparse::sparse_linear_vectorized_grouped_parallel_forward_wrapper,
        "AVX512 implementation of grouped parallel sparse forward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_grouped_backward",
        &sparse::sparse_linear_vectorized_grouped_backward_wrapper,
        "AVX512 implementation of grouped sparse backward propagation algorithm for linear layers."
    );
    m.def(
        "sparse_linear_vectorized_grouped_parallel_backward",
        &sparse::sparse_linear_vectorized_grouped_parallel_backward_wrapper,
        "AVX512 implementation of grouped parallel sparse backward propagation algorithm for linear layers."
    );

    // Conv2d - stride 1
    m.def(
        "sparse_conv2d_vectorized_forward_stride_1",
        &sparse::sparse_conv2d_vectorized_forward_stride_1_wrapper,
        "AVX512 implementation of sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_parallel_forward_stride_1",
        &sparse::sparse_conv2d_vectorized_parallel_forward_stride_1_wrapper,
        "AVX512 implementation of parallel sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_backward_stride_1",
        &sparse::sparse_conv2d_vectorized_backward_stride_1_wrapper,
        "AVX512 implementation of sparse backward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_parallel_backward_stride_1",
        &sparse::sparse_conv2d_vectorized_parallel_backward_stride_1_wrapper,
        "AVX512 implementation of parallel sparse backward propagation algorithm for convolutional layers."
    );

    m.def(
        "sparse_conv2d_vectorized_grouped_forward_stride_1",
        &sparse::sparse_conv2d_vectorized_grouped_forward_stride_1_wrapper,
        "AVX512 implementation of grouped sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_parallel_forward_stride_1",
        &sparse::sparse_conv2d_vectorized_grouped_parallel_forward_stride_1_wrapper,
        "AVX512 implementation of grouped parallel sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_backward_stride_1",
        &sparse::sparse_conv2d_vectorized_grouped_backward_stride_1_wrapper,
        "AVX512 implementation of grouped sparse backward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_parallel_backward_stride_1",
        &sparse::sparse_conv2d_vectorized_grouped_parallel_backward_stride_1_wrapper,
        "AVX512 implementation of grouped parallel sparse backward propagation algorithm for convolutional layers."
    );

    // Conv2d - stride 2
    m.def(
        "sparse_conv2d_vectorized_forward_stride_2",
        &sparse::sparse_conv2d_vectorized_forward_stride_2_wrapper,
        "AVX512 implementation of sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_parallel_forward_stride_2",
        &sparse::sparse_conv2d_vectorized_parallel_forward_stride_2_wrapper,
        "AVX512 implementation of parallel sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_backward_stride_2",
        &sparse::sparse_conv2d_vectorized_backward_stride_2_wrapper,
        "AVX512 implementation of sparse backward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_parallel_backward_stride_2",
        &sparse::sparse_conv2d_vectorized_parallel_backward_stride_2_wrapper,
        "AVX512 implementation of parallel sparse backward propagation algorithm for convolutional layers."
    );

    m.def(
        "sparse_conv2d_vectorized_grouped_forward_stride_2",
        &sparse::sparse_conv2d_vectorized_grouped_forward_stride_2_wrapper,
        "AVX512 implementation of grouped sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_parallel_forward_stride_2",
        &sparse::sparse_conv2d_vectorized_grouped_parallel_forward_stride_2_wrapper,
        "AVX512 implementation of grouped parallel sparse forward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_backward_stride_2",
        &sparse::sparse_conv2d_vectorized_grouped_backward_stride_2_wrapper,
        "AVX512 implementation of grouped sparse backward propagation algorithm for convolutional layers."
    );
    m.def(
        "sparse_conv2d_vectorized_grouped_parallel_backward_stride_2",
        &sparse::sparse_conv2d_vectorized_grouped_parallel_backward_stride_2_wrapper,
        "AVX512 implementation of grouped parallel sparse backward propagation algorithm for convolutional layers."
    );

    // Conv2d Utils
    m.def(
        "sparsify_conv2d",
        &sparse::sparsify_conv2d_wrapper,
        "Utility function to convert convolutional weights into sparse format as specified in the SparseProp paper."
    );
    m.def(
        "densify_conv2d",
        &sparse::densify_conv2d_wrapper,
        "Utility function to convert sparse convolutional weights into dense format."
    );

    // Quantization Strategies
    m.def(
        "dithered_8_bit_scalar_quantization",
        &quantization::dithered_8_bit_scalar_quantization_wrapper,
        "AVX512 implementation of 8 bit dithered quantization strategy."
    );
    m.def(
        "dithered_8_bit_quantization",
        &quantization::dithered_8_bit_quantization_wrapper,
        "AVX512 implementation of 8 bit dithered quantization strategy."
    );
    m.def(
        "dithered_8_bit_quantization_parallel",
        &quantization::dithered_8_bit_quantization_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit dithered quantization strategy."
    );
    m.def(
        "dithered_8_bit_quantization_grouped",
        &quantization::dithered_8_bit_quantization_grouped_wrapper,
        "AVX512 implementation of 8 bit grouped dithered quantization strategy."
    );
    m.def(
        "dithered_8_bit_quantization_grouped_parallel",
        &quantization::dithered_8_bit_quantization_grouped_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit grouped dithered quantization strategy."
    );

    m.def(
        "sawb_8_bit_scalar_quantization",
        &quantization::sawb_8_bit_scalar_quantization_wrapper,
        "AVX512 implementation of 8 bit SAWB quantization strategy."
    );
    m.def(
        "sawb_8_bit_quantization",
        &quantization::sawb_8_bit_quantization_wrapper,
        "AVX512 implementation of 8 bit SAWB quantization strategy."
    );
    m.def(
        "sawb_8_bit_quantization_parallel",
        &quantization::sawb_8_bit_quantization_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit SAWB quantization strategy."
    );
    m.def(
        "sawb_8_bit_quantization_grouped",
        &quantization::sawb_8_bit_quantization_grouped_wrapper,
        "AVX512 implementation of 8 bit grouped SAWB quantization strategy."
    );
    m.def(
        "sawb_8_bit_quantization_grouped_parallel",
        &quantization::sawb_8_bit_quantization_grouped_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit grouped SAWB quantization strategy."
    );

    m.def(
        "standard_8_bit_quantization",
        &quantization::standard_8_bit_quantization_wrapper,
        "AVX512 implementation of 8 bit standard quantization strategy."
    );
    m.def(
        "standard_8_bit_quantization_parallel",
        &quantization::standard_8_bit_quantization_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit standard quantization strategy."
    );
    m.def(
        "standard_8_bit_quantization_grouped",
        &quantization::standard_8_bit_quantization_grouped_wrapper,
        "AVX512 implementation of 8 bit standard quantization strategy."
    );
    m.def(
        "standard_8_bit_quantization_grouped_parallel",
        &quantization::standard_8_bit_quantization_grouped_parallel_wrapper,
        "Parallel AVX512 implementation of 8 bit standard quantization strategy."
    );

    // Tensor Utils
    m.def("transpose", &transpose_wrapper, "AVX512 implementation of parallel transpose operation.");

    // Sparse Kernels
    // linear
    m.def("uq_sparse_linear_vectorized_forward", &sparse::uq_sparse_linear_vectorized_forward_wrapper);
    m.def("uq_sparse_linear_vectorized_backward", &sparse::uq_sparse_linear_vectorized_backward_wrapper);

    // conv2d
    m.def("uq_sparse_conv2d_vectorized_forward_stride_1", &sparse::uq_sparse_conv2d_vectorized_forward_stride_1_wrapper);
    m.def("uq_sparse_conv2d_vectorized_backward_stride_1", &sparse::uq_sparse_conv2d_vectorized_backward_stride_1_wrapper);
    m.def("uq_sparse_conv2d_vectorized_forward_stride_2", &sparse::uq_sparse_conv2d_vectorized_forward_stride_2_wrapper);
    m.def("uq_sparse_conv2d_vectorized_backward_stride_2", &sparse::uq_sparse_conv2d_vectorized_backward_stride_2_wrapper);

    // conv2d over on
    m.def("uq_sparse_conv2d_vectorized_forward_over_on_stride_1", &sparse::uq_sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper);
    m.def("uq_sparse_conv2d_vectorized_backward_over_on_stride_1", &sparse::uq_sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper);
    m.def("uq_sparse_conv2d_vectorized_backward_over_on_stride_2", &sparse::uq_sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper);

    m.def("sparse_conv2d_vectorized_forward_over_on_stride_1", &sparse::sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_1", &sparse::sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_2", &sparse::sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper);
    m.def("sparse_conv2d_vectorized_parallel_forward_over_on_stride_1", &sparse::sparse_conv2d_vectorized_parallel_forward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_parallel_backward_over_on_stride_1", &sparse::sparse_conv2d_vectorized_parallel_backward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_parallel_backward_over_on_stride_2", &sparse::sparse_conv2d_vectorized_parallel_backward_over_on_stride_2_wrapper);
}