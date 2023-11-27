#include "qsparse_linear.h"

#include <iostream>

void sparse::quantized_sparse_linear_vectorized_forward(
    int B, int M, int N, int8_t* X, float X_scale,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale, float* O
) {
    for(int i = 0; i < N; i++) {
        const int wcol_idx_start = W_idx_N[i];
        const int wcol_idx_end = W_idx_N[i + 1];

        for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
            const int wrow = W_idx_M[j];
            const float weight = (float) W_val[j] * W_scale;
            const float w_mul_x_scale = weight * X_scale;
            const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);

            int idx;
            const int offset_0 = wrow * B;
            const int offset_1 = i * B;
            int8_t* X_ptr = X + offset_0;
            float* O_ptr = O + offset_1;
            for (idx = 0; idx < B - 63; idx += 64) {
                float* O_base = O_ptr + idx;
                const __m512 o_0 = _mm512_loadu_ps(O_base);
                const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                int8_t* X_base = X_ptr + idx;
                const __m256i x_0 = _mm256_loadu_epi32(X_base);
                const __m256i x_1 = _mm256_loadu_epi32(X_base + 32);
                __m256i x_unpack_0_256;
                __m256i x_unpack_1_256;
                __m256i x_unpack_2_256;
                __m256i x_unpack_3_256;
                __m256i x_unpack_4_256;
                __m256i x_unpack_5_256;
                __m256i x_unpack_6_256;
                __m256i x_unpack_7_256;
                quantization::Quantization8Strategy::unpack64(
                    x_0, x_1,
                    x_unpack_0_256, x_unpack_1_256, x_unpack_2_256, x_unpack_3_256,
                    x_unpack_4_256, x_unpack_5_256, x_unpack_6_256, x_unpack_7_256
                );
                const __m512i x_unpack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_0_256), x_unpack_1_256, 1);
                const __m512i x_unpack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_2_256), x_unpack_3_256, 1);
                const __m512i x_unpack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_4_256), x_unpack_5_256, 1);
                const __m512i x_unpack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_6_256), x_unpack_7_256, 1);

                const __m512 x_f_0_512 = _mm512_cvtepi32_ps(x_unpack_0_512);
                const __m512 x_f_1_512 = _mm512_cvtepi32_ps(x_unpack_1_512);
                const __m512 x_f_2_512 = _mm512_cvtepi32_ps(x_unpack_2_512);
                const __m512 x_f_3_512 = _mm512_cvtepi32_ps(x_unpack_3_512);

                const __m512 r0 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_0_512, o_0);
                const __m512 r1 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_1_512, o_1);
                const __m512 r2 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_2_512, o_2);
                const __m512 r3 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_3_512, o_3);

                _mm512_storeu_ps(O_base, r0);
                _mm512_storeu_ps(O_base + 16, r1);
                _mm512_storeu_ps(O_base + 32, r2);
                _mm512_storeu_ps(O_base + 48, r3);
            }

            for (; idx < B; idx++) {
                const float o = O_ptr[idx];
                const float x = (float) X_ptr[idx];

                const float x_mul_w = x * w_mul_x_scale;
                const float o_update = o + x_mul_w;

                O_ptr[idx] = o_update;
            }
        }
    }
}

void sparse::quantized_sparse_linear_vectorized_parallel_forward(
    int B, int M, int N, int8_t* X, float X_scale,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale, float* O
) {
#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(B, N, X, O, X_scale, W_idx_N, W_idx_M, W_val, W_scale)
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            const int wcol_idx_start = W_idx_N[i];
            const int wcol_idx_end = W_idx_N[i + 1];

            for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
                const int wrow = W_idx_M[j];
                const float weight = (float) W_val[j] * W_scale;
                const float w_mul_x_scale = weight * X_scale;
                const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);

                int idx;
                const int offset_0 = wrow * B;
                const int offset_1 = i * B;
                int8_t *X_ptr = X + offset_0;
                float *O_ptr = O + offset_1;
                for (idx = 0; idx < B - 63; idx += 64) {
                    float *O_base = O_ptr + idx;
                    const __m512 o_0 = _mm512_loadu_ps(O_base);
                    const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                    const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                    const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                    int8_t *X_base = X_ptr + idx;
                    const __m256i x_0 = _mm256_loadu_epi32(X_base);
                    const __m256i x_1 = _mm256_loadu_epi32(X_base + 32);
                    __m256i x_unpack_0_256;
                    __m256i x_unpack_1_256;
                    __m256i x_unpack_2_256;
                    __m256i x_unpack_3_256;
                    __m256i x_unpack_4_256;
                    __m256i x_unpack_5_256;
                    __m256i x_unpack_6_256;
                    __m256i x_unpack_7_256;
                    quantization::Quantization8Strategy::unpack64(
                        x_0, x_1,
                        x_unpack_0_256, x_unpack_1_256, x_unpack_2_256, x_unpack_3_256,
                        x_unpack_4_256, x_unpack_5_256, x_unpack_6_256, x_unpack_7_256
                    );
                    const __m512i x_unpack_0_512 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(x_unpack_0_256), x_unpack_1_256, 1
                    );
                    const __m512i x_unpack_1_512 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(x_unpack_2_256), x_unpack_3_256, 1
                    );
                    const __m512i x_unpack_2_512 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(x_unpack_4_256), x_unpack_5_256, 1
                    );
                    const __m512i x_unpack_3_512 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(x_unpack_6_256), x_unpack_7_256, 1
                    );

                    const __m512 x_f_0_512 = _mm512_cvtepi32_ps(x_unpack_0_512);
                    const __m512 x_f_1_512 = _mm512_cvtepi32_ps(x_unpack_1_512);
                    const __m512 x_f_2_512 = _mm512_cvtepi32_ps(x_unpack_2_512);
                    const __m512 x_f_3_512 = _mm512_cvtepi32_ps(x_unpack_3_512);

                    const __m512 r0 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_0_512, o_0);
                    const __m512 r1 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_1_512, o_1);
                    const __m512 r2 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_2_512, o_2);
                    const __m512 r3 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_3_512, o_3);

                    _mm512_storeu_ps(O_base, r0);
                    _mm512_storeu_ps(O_base + 16, r1);
                    _mm512_storeu_ps(O_base + 32, r2);
                    _mm512_storeu_ps(O_base + 48, r3);
                }

                for (; idx < B; idx++) {
                    const float o = O_ptr[idx];
                    const float x = (float) X_ptr[idx];

                    const float x_mul_w = x * w_mul_x_scale;
                    const float o_update = o + x_mul_w;

                    O_ptr[idx] = o_update;
                }
            }
        }
    }
#else
    sparse::quantized_sparse_linear_vectorized_forward(B, M, N, X, X_scale, W_idx_N, W_idx_M, W_val, W_scale, O);
#endif
}

void sparse::quantized_grouped_sparse_linear_vectorized_forward(
    int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
) {
    for(int i = 0; i < N; i++) {
        const int wcol_idx_start = W_idx_N[i];
        const int wcol_idx_end = W_idx_N[i + 1];

        for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
            const int wrow = W_idx_M[j];
            const int W_group = j / W_qgroup_size;
            const float weight = (float) W_val[j] * W_scale[W_group];

            int idx;
            const int offset_0 = wrow * B;
            const int offset_1 = i * B;
            int8_t* X_ptr = X + offset_0;
            float* O_ptr = O + offset_1;
            for (idx = 0; idx < B - 63; idx += 64) {
                float* O_base = O_ptr + idx;
                const __m512 o_0 = _mm512_loadu_ps(O_base);
                const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                int8_t* X_base = X_ptr + idx;
                const __m256i x_0 = _mm256_loadu_epi32(X_base);
                const __m256i x_1 = _mm256_loadu_epi32(X_base + 32);
                __m256i x_unpack_0_256;
                __m256i x_unpack_1_256;
                __m256i x_unpack_2_256;
                __m256i x_unpack_3_256;
                __m256i x_unpack_4_256;
                __m256i x_unpack_5_256;
                __m256i x_unpack_6_256;
                __m256i x_unpack_7_256;
                quantization::Quantization8Strategy::unpack64(
                    x_0, x_1,
                    x_unpack_0_256, x_unpack_1_256, x_unpack_2_256, x_unpack_3_256,
                    x_unpack_4_256, x_unpack_5_256, x_unpack_6_256, x_unpack_7_256
                );
                const __m512i x_unpack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_0_256), x_unpack_1_256, 1);
                const __m512i x_unpack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_2_256), x_unpack_3_256, 1);
                const __m512i x_unpack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_4_256), x_unpack_5_256, 1);
                const __m512i x_unpack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_6_256), x_unpack_7_256, 1);

                const __m512 x_f_0_512 = _mm512_cvtepi32_ps(x_unpack_0_512);
                const __m512 x_f_1_512 = _mm512_cvtepi32_ps(x_unpack_1_512);
                const __m512 x_f_2_512 = _mm512_cvtepi32_ps(x_unpack_2_512);
                const __m512 x_f_3_512 = _mm512_cvtepi32_ps(x_unpack_3_512);

                // For this to work, the group size for X needs to be at least 64 because of the loop unrolling & vectorization
                const int X_group = (offset_0 + idx) / X_qgroup_size;
                const float w_mul_x_scale = weight * X_scale[X_group];
                const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);
                const __m512 r0 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_0_512, o_0);
                const __m512 r1 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_1_512, o_1);
                const __m512 r2 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_2_512, o_2);
                const __m512 r3 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_3_512, o_3);

                _mm512_storeu_ps(O_base, r0);
                _mm512_storeu_ps(O_base + 16, r1);
                _mm512_storeu_ps(O_base + 32, r2);
                _mm512_storeu_ps(O_base + 48, r3);
            }

            for (; idx < B; idx++) {
                const float o = O_ptr[idx];
                const float x = (float) X_ptr[idx];

                const int X_group = (offset_0 + idx) / X_qgroup_size;
                const float w_mul_x_scale = weight * X_scale[X_group];

                const float x_mul_w = x * w_mul_x_scale;
                const float o_update = o + x_mul_w;

                O_ptr[idx] = o_update;
            }
        }
    }
}

void sparse::quantized_grouped_sparse_linear_vectorized_parallel_forward(
    int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
) {
#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(B, N, X, O, X_scale, W_idx_N, W_idx_M, W_val, W_scale, W_qgroup_size, X_qgroup_size)
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            const int wcol_idx_start = W_idx_N[i];
            const int wcol_idx_end = W_idx_N[i + 1];

            for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
                const int wrow = W_idx_M[j];
                const int W_group = j / W_qgroup_size;
                const float weight = (float) W_val[j] * W_scale[W_group];

                int idx;
                const int offset_0 = wrow * B;
                const int offset_1 = i * B;
                int8_t *X_ptr = X + offset_0;
                float *O_ptr = O + offset_1;
                for (idx = 0; idx < B - 63; idx += 64) {
                    float *O_base = O_ptr + idx;
                    const __m512 o_0 = _mm512_loadu_ps(O_base);
                    const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                    const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                    const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                    int8_t *X_base = X_ptr + idx;
                    const __m256i x_0 = _mm256_loadu_epi32(X_base);
                    const __m256i x_1 = _mm256_loadu_epi32(X_base + 32);
                    __m256i x_unpack_0_256;
                    __m256i x_unpack_1_256;
                    __m256i x_unpack_2_256;
                    __m256i x_unpack_3_256;
                    __m256i x_unpack_4_256;
                    __m256i x_unpack_5_256;
                    __m256i x_unpack_6_256;
                    __m256i x_unpack_7_256;
                    quantization::Quantization8Strategy::unpack64(
                        x_0, x_1,
                        x_unpack_0_256, x_unpack_1_256, x_unpack_2_256, x_unpack_3_256,
                        x_unpack_4_256, x_unpack_5_256, x_unpack_6_256, x_unpack_7_256
                    );
                    const __m512i x_unpack_0_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_0_256), x_unpack_1_256, 1);
                    const __m512i x_unpack_1_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_2_256), x_unpack_3_256, 1);
                    const __m512i x_unpack_2_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_4_256), x_unpack_5_256, 1);
                    const __m512i x_unpack_3_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(x_unpack_6_256), x_unpack_7_256, 1);

                    const __m512 x_f_0_512 = _mm512_cvtepi32_ps(x_unpack_0_512);
                    const __m512 x_f_1_512 = _mm512_cvtepi32_ps(x_unpack_1_512);
                    const __m512 x_f_2_512 = _mm512_cvtepi32_ps(x_unpack_2_512);
                    const __m512 x_f_3_512 = _mm512_cvtepi32_ps(x_unpack_3_512);

                    // For this to work, the group size for X needs to be at least 64 because of the loop unrolling & vectorization
                    const int X_group = (offset_0 + idx) / X_qgroup_size;
                    const float w_mul_x_scale = weight * X_scale[X_group];
                    const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);
                    const __m512 r0 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_0_512, o_0);
                    const __m512 r1 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_1_512, o_1);
                    const __m512 r2 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_2_512, o_2);
                    const __m512 r3 = _mm512_fmadd_ps(w_mul_x_scale_vec, x_f_3_512, o_3);

                    _mm512_storeu_ps(O_base, r0);
                    _mm512_storeu_ps(O_base + 16, r1);
                    _mm512_storeu_ps(O_base + 32, r2);
                    _mm512_storeu_ps(O_base + 48, r3);
                }

                for (; idx < B; idx++) {
                    const float o = O_ptr[idx];
                    const float x = (float) X_ptr[idx];

                    const int X_group = (offset_0 + idx) / X_qgroup_size;
                    const float w_mul_x_scale = weight * X_scale[X_group];

                    const float x_mul_w = x * w_mul_x_scale;
                    const float o_update = o + x_mul_w;

                    O_ptr[idx] = o_update;
                }
            }
        }
    }
#else
    sparse::quantized_grouped_sparse_linear_vectorized_forward(B, M, N, X, X_scale, X_qgroup_size, W_idx_N, W_idx_M, W_val, W_scale, W_qgroup_size, O);
#endif
}

// We use the transpose of the weight vector, thus it is stored in CSC format.
void sparse::quantized_sparse_linear_vectorized_backward(
    int B, int M, int N, int8_t* X, float X_scale,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale,
    int8_t* dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
) {
    for(int i = 0; i < N; i++) {
        const int wcol_idx_start = W_idx_N[i];
        const int wcol_idx_end = W_idx_N[i + 1];
        for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
            const int wrow = W_idx_M[j];
            const float weight = (float) W_val[j] * W_scale;
            const float w_mul_dLdO_scale = weight * dLdO_scale;
            const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
            const float X_mul_dLdO_scale = X_scale * dLdO_scale;
            const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
            int s_acc_i = 0;
            __m512i acc_vec_i = _mm512_setzero_epi32();

            const int offset_0 = wrow * B;
            const int offset_1 = i * B;
            float* dLdX_ptr = dLdX + offset_0;
            int8_t* X_ptr = X + offset_0;
            int8_t* dLdO_ptr = dLdO + offset_1;
            int idx;
            for (idx = 0; idx < B - 63; idx += 64) {
                float* dLdX_base = dLdX_ptr + idx;
                const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);
                const __m512i x0 = _mm512_loadu_epi32(X_ptr + idx);
                const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + idx);

                // Unpack do0
                const __m256i do0_lo = _mm512_castsi512_si256(do0); // Cast has 0 latency!!!
                const __m256i do0_hi = _mm512_extracti32x8_epi32(do0, 0x1);
                __m256i do0_unpack_0_256;
                __m256i do0_unpack_1_256;
                __m256i do0_unpack_2_256;
                __m256i do0_unpack_3_256;
                __m256i do0_unpack_4_256;
                __m256i do0_unpack_5_256;
                __m256i do0_unpack_6_256;
                __m256i do0_unpack_7_256;
                quantization::Quantization8Strategy::unpack64(
                    do0_lo, do0_hi,
                    do0_unpack_0_256, do0_unpack_1_256, do0_unpack_2_256, do0_unpack_3_256,
                    do0_unpack_4_256, do0_unpack_5_256, do0_unpack_6_256, do0_unpack_7_256
                );
                // Combine 256 bit vectors into 512 bit vectors to speed up following operations
                const __m512i do0_unpack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_0_256), do0_unpack_1_256, 1);
                const __m512i do0_unpack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_2_256), do0_unpack_3_256, 1);
                const __m512i do0_unpack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_4_256), do0_unpack_5_256, 1);
                const __m512i do0_unpack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_6_256), do0_unpack_7_256, 1);

                const __m512 do0_f_0_512 = _mm512_cvtepi32_ps(do0_unpack_0_512);
                const __m512 do0_f_1_512 = _mm512_cvtepi32_ps(do0_unpack_1_512);
                const __m512 do0_f_2_512 = _mm512_cvtepi32_ps(do0_unpack_2_512);
                const __m512 do0_f_3_512 = _mm512_cvtepi32_ps(do0_unpack_3_512);

                // Compute dLdX updates
                const __m512 s0 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_0_512, dx0);
                const __m512 s1 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_1_512, dx1);
                const __m512 s2 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_2_512, dx2);
                const __m512 s3 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_3_512, dx3);

                // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by its absolute value
                // We use x0's sign to calculate the overall sign of the multiplication, and assign that sign to
                __m512i x0_signs = _mm512_and_si512(x0, _mm512_1st_bit_on_epi8);
                __m512i x0_abs = _mm512_abs_epi8(x0);
                __m512i do0_abs = _mm512_abs_epi8(do0);
                __m512i do0_signs = _mm512_and_si512(do0, _mm512_1st_bit_on_epi8);
                __mmask64 mask = _mm512_cmp_epi8_mask(x0_signs, do0_signs, _MM_CMPINT_NE);
                __m512i do0_signed  = _mm512_mask_sub_epi8(do0_abs, mask, _mm512_zeros_epi32, do0_abs);
                acc_vec_i = _mm512_dpbusd_epi32(acc_vec_i, x0_abs, do0_signed);

                _mm512_storeu_ps(dLdX_base, s0);
                _mm512_storeu_ps(dLdX_base + 16, s1);
                _mm512_storeu_ps(dLdX_base + 32, s2);
                _mm512_storeu_ps(dLdX_base + 48, s3);
            }
            const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
            const __m512 acc_vec = _mm512_mul_ps(X_mul_dLdO_scale_vec, acc_vec_f);
            const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

            for (; idx < B; idx++) {
                const float dx0 = dLdX_ptr[idx];
                const int x0 = (int) X_ptr[idx];
                const int do0 = (int) dLdO_ptr[idx];

                const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                const float dx_update = dx0 + do0_mul_w;
                const int x0_mul_do0 = x0 * do0;

                s_acc_i = s_acc_i + x0_mul_do0;
                dLdX_ptr[idx] = dx_update;
            }
            const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

            dLdW_val[j] = s_acc_0 + s_acc_1;
        }
    }
}

void sparse::quantized_sparse_linear_vectorized_parallel_backward(
    int B, int M, int N, int8_t* X, float X_scale,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float W_scale,
    int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(B, N, X, X_scale, dLdO, dLdO_scale, W_idx_N, W_idx_M, W_val, W_scale, dLdX, dLdW_val, _mm512_1st_bit_on_epi8, _mm512_zeros_epi32)
    {
        for(int i = 0; i < N; i++) {
            const int wcol_idx_start = W_idx_N[i];
            const int wcol_idx_end = W_idx_N[i + 1];
            #pragma omp for
            for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
                const int wrow = W_idx_M[j];
                const float weight = (float) W_val[j] * W_scale;
                const float w_mul_dLdO_scale = weight * dLdO_scale;
                const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                int s_acc_i = 0;
                __m512i acc_vec_i = _mm512_setzero_epi32();

                const int offset_0 = wrow * B;
                const int offset_1 = i * B;
                float* dLdX_ptr = dLdX + offset_0;
                int8_t* X_ptr = X + offset_0;
                int8_t* dLdO_ptr = dLdO + offset_1;
                int idx;
                for (idx = 0; idx < B - 63; idx += 64) {
                    float* dLdX_base = dLdX_ptr + idx;
                    const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                    const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                    const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                    const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);
                    const __m512i x0 = _mm512_loadu_epi32(X_ptr + idx);
                    const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + idx);

                    // Unpack do0
                    const __m256i do0_lo = _mm512_castsi512_si256(do0); // Cast has 0 latency!!!
                    const __m256i do0_hi = _mm512_extracti32x8_epi32(do0, 0x1);
                    __m256i do0_unpack_0_256;
                    __m256i do0_unpack_1_256;
                    __m256i do0_unpack_2_256;
                    __m256i do0_unpack_3_256;
                    __m256i do0_unpack_4_256;
                    __m256i do0_unpack_5_256;
                    __m256i do0_unpack_6_256;
                    __m256i do0_unpack_7_256;
                    quantization::Quantization8Strategy::unpack64(
                        do0_lo, do0_hi,
                        do0_unpack_0_256, do0_unpack_1_256, do0_unpack_2_256, do0_unpack_3_256,
                        do0_unpack_4_256, do0_unpack_5_256, do0_unpack_6_256, do0_unpack_7_256
                    );
                    // Combine 256 bit vectors into 512 bit vectors to speed up following operations
                    const __m512i do0_unpack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_0_256), do0_unpack_1_256, 1);
                    const __m512i do0_unpack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_2_256), do0_unpack_3_256, 1);
                    const __m512i do0_unpack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_4_256), do0_unpack_5_256, 1);
                    const __m512i do0_unpack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_6_256), do0_unpack_7_256, 1);

                    const __m512 do0_f_0_512 = _mm512_cvtepi32_ps(do0_unpack_0_512);
                    const __m512 do0_f_1_512 = _mm512_cvtepi32_ps(do0_unpack_1_512);
                    const __m512 do0_f_2_512 = _mm512_cvtepi32_ps(do0_unpack_2_512);
                    const __m512 do0_f_3_512 = _mm512_cvtepi32_ps(do0_unpack_3_512);

                    // Compute dLdX updates
                    const __m512 s0 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_0_512, dx0);
                    const __m512 s1 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_1_512, dx1);
                    const __m512 s2 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_2_512, dx2);
                    const __m512 s3 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_3_512, dx3);

                    // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by its absolute value
                    // We use x0's sign to calculate the overall sign of the multiplication, and assign that sign to
                    __m512i x0_signs = _mm512_and_si512(x0, _mm512_1st_bit_on_epi8);
                    __m512i x0_abs = _mm512_abs_epi8(x0);
                    __m512i do0_abs = _mm512_abs_epi8(do0);
                    __m512i do0_signs = _mm512_and_si512(do0, _mm512_1st_bit_on_epi8);
                    __mmask64 mask = _mm512_cmp_epi8_mask(x0_signs, do0_signs, _MM_CMPINT_NE);
                    __m512i do0_signed  = _mm512_mask_sub_epi8(do0_abs, mask, _mm512_zeros_epi32, do0_abs);
                    acc_vec_i = _mm512_dpbusd_epi32(acc_vec_i, x0_abs, do0_signed);

                    _mm512_storeu_ps(dLdX_base, s0);
                    _mm512_storeu_ps(dLdX_base + 16, s1);
                    _mm512_storeu_ps(dLdX_base + 32, s2);
                    _mm512_storeu_ps(dLdX_base + 48, s3);
                }
                const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                const __m512 acc_vec = _mm512_mul_ps(X_mul_dLdO_scale_vec, acc_vec_f);
                const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

                for (; idx < B; idx++) {
                    const float dx0 = dLdX_ptr[idx];
                    const int x0 = (int) X_ptr[idx];
                    const int do0 = (int) dLdO_ptr[idx];

                    const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                    const float dx_update = dx0 + do0_mul_w;
                    const int x0_mul_do0 = x0 * do0;

                    s_acc_i = s_acc_i + x0_mul_do0;
                    dLdX_ptr[idx] = dx_update;
                }
                const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

                dLdW_val[j] = s_acc_0 + s_acc_1;
            }
        }
    }
#else
    sparse::quantized_sparse_linear_vectorized_backward(
        B, M, N, X, X_scale, W_idx_N, W_idx_M, W_val, W_scale, dLdO, dLdO_scale, dLdX, dLdW_val
    );
#endif
}

void sparse::quantized_grouped_sparse_linear_vectorized_backward(
    int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size,
    int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
) {
    for(int i = 0; i < N; i++) {
        const int wcol_idx_start = W_idx_N[i];
        const int wcol_idx_end = W_idx_N[i + 1];
        for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
            const int wrow = W_idx_M[j];
            const int W_group = j / W_qgroup_size;
            const float weight = (float) W_val[j] * W_scale[W_group];

            float s_acc_1 = 0.0f;
            __m512 acc_vec = _mm512_setzero_ps();

            const int offset_0 = wrow * B;
            const int offset_1 = i * B;
            float* dLdX_ptr = dLdX + offset_0;
            int8_t* X_ptr = X + offset_0;
            int8_t* dLdO_ptr = dLdO + offset_1;
            int idx;
            for (idx = 0; idx < B - 63; idx += 64) {
                float* dLdX_base = dLdX_ptr + idx;
                const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);
                const __m512i x0 = _mm512_loadu_epi32(X_ptr + idx);
                const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + idx);

                // Unpack do0
                const __m256i do0_lo = _mm512_castsi512_si256(do0); // Cast has 0 latency!!!
                const __m256i do0_hi = _mm512_extracti32x8_epi32(do0, 0x1);
                __m256i do0_unpack_0_256;
                __m256i do0_unpack_1_256;
                __m256i do0_unpack_2_256;
                __m256i do0_unpack_3_256;
                __m256i do0_unpack_4_256;
                __m256i do0_unpack_5_256;
                __m256i do0_unpack_6_256;
                __m256i do0_unpack_7_256;
                quantization::Quantization8Strategy::unpack64(
                    do0_lo, do0_hi,
                    do0_unpack_0_256, do0_unpack_1_256, do0_unpack_2_256, do0_unpack_3_256,
                    do0_unpack_4_256, do0_unpack_5_256, do0_unpack_6_256, do0_unpack_7_256
                );
                // Combine 256 bit vectors into 512 bit vectors to speed up following operations
                const __m512i do0_unpack_0_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_0_256), do0_unpack_1_256, 1);
                const __m512i do0_unpack_1_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_2_256), do0_unpack_3_256, 1);
                const __m512i do0_unpack_2_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_4_256), do0_unpack_5_256, 1);
                const __m512i do0_unpack_3_512 = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_6_256), do0_unpack_7_256, 1);

                const __m512 do0_f_0_512 = _mm512_cvtepi32_ps(do0_unpack_0_512);
                const __m512 do0_f_1_512 = _mm512_cvtepi32_ps(do0_unpack_1_512);
                const __m512 do0_f_2_512 = _mm512_cvtepi32_ps(do0_unpack_2_512);
                const __m512 do0_f_3_512 = _mm512_cvtepi32_ps(do0_unpack_3_512);

                // Compute dLdX updates
                const int dLdO_group = (offset_1 + idx) / dLdO_qgroup_size;
                const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                const __m512 s0 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_0_512, dx0);
                const __m512 s1 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_1_512, dx1);
                const __m512 s2 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_2_512, dx2);
                const __m512 s3 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_3_512, dx3);

                // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by its absolute value
                // We use x0's sign to calculate the overall sign of the multiplication, and assign that sign to
                __m512i x0_signs = _mm512_and_si512(x0, _mm512_1st_bit_on_epi8);
                __m512i x0_abs = _mm512_abs_epi8(x0);
                __m512i do0_abs = _mm512_abs_epi8(do0);
                __m512i do0_signs = _mm512_and_si512(do0, _mm512_1st_bit_on_epi8);
                __mmask64 mask = _mm512_cmp_epi8_mask(x0_signs, do0_signs, _MM_CMPINT_NE);
                __m512i do0_signed  = _mm512_mask_sub_epi8(do0_abs, mask, _mm512_zeros_epi32, do0_abs);
                const __m512i acc_vec_i = _mm512_dpbusd_epi32(_mm512_zeros_epi32, x0_abs, do0_signed);
                const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);

                const int X_group = (offset_0 + idx) / X_qgroup_size;
                const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];
                const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                acc_vec = _mm512_fmadd_ps(acc_vec_f, X_mul_dLdO_scale_vec, acc_vec);

                _mm512_storeu_ps(dLdX_base, s0);
                _mm512_storeu_ps(dLdX_base + 16, s1);
                _mm512_storeu_ps(dLdX_base + 32, s2);
                _mm512_storeu_ps(dLdX_base + 48, s3);
            }
            const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

            for (; idx < B; idx++) {
                const float dx0 = dLdX_ptr[idx];
                const int x0 = (int) X_ptr[idx];
                const int do0 = (int) dLdO_ptr[idx];

                const int dLdO_group = (offset_1 + idx) / dLdO_qgroup_size;
                const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                const int X_group = (offset_0 + idx) / X_qgroup_size;
                const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];

                const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                const float dx_update = dx0 + do0_mul_w;
                const int x0_mul_do0 = x0 * do0;

                const float s_acc_f = X_mul_dLdO_scale * x0_mul_do0;
                s_acc_1 += s_acc_f;
                dLdX_ptr[idx] = dx_update;
            }

            dLdW_val[j] = s_acc_0 + s_acc_1;
        }
    }
}

void sparse::quantized_grouped_sparse_linear_vectorized_parallel_backward(
    int B, int M, int N, int8_t* X, float* X_scale, int X_qgroup_size,
    int* W_idx_N, int* W_idx_M, int8_t* W_val, float* W_scale, int W_qgroup_size,
    int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(B, N, X, X_scale, dLdO, dLdO_scale, W_idx_N, W_idx_M, W_val, W_scale, dLdX, dLdW_val, _mm512_1st_bit_on_epi8, _mm512_zeros_epi32, X_qgroup_size, W_qgroup_size, dLdO_qgroup_size)
    {
        for (int i = 0; i < N; i++) {
            const int wcol_idx_start = W_idx_N[i];
            const int wcol_idx_end = W_idx_N[i + 1];

            #pragma omp for
            for (int j = wcol_idx_start; j < wcol_idx_end; j++) {
                const int wrow = W_idx_M[j];
                const int W_group = j / W_qgroup_size;
                const float weight = (float) W_val[j] * W_scale[W_group];

                float s_acc_1 = 0.0f;
                __m512 acc_vec = _mm512_setzero_ps();

                const int offset_0 = wrow * B;
                const int offset_1 = i * B;
                float *dLdX_ptr = dLdX + offset_0;
                int8_t *X_ptr = X + offset_0;
                int8_t *dLdO_ptr = dLdO + offset_1;
                int idx;
                for (idx = 0; idx < B - 63; idx += 64) {
                    float *dLdX_base = dLdX_ptr + idx;
                    const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                    const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                    const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                    const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);
                    const __m512i x0 = _mm512_loadu_epi32(X_ptr + idx);
                    const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + idx);

                    // Unpack do0
                    const __m256i do0_lo = _mm512_castsi512_si256(do0); // Cast has 0 latency!!!
                    const __m256i do0_hi = _mm512_extracti32x8_epi32(do0, 0x1);
                    __m256i do0_unpack_0_256;
                    __m256i do0_unpack_1_256;
                    __m256i do0_unpack_2_256;
                    __m256i do0_unpack_3_256;
                    __m256i do0_unpack_4_256;
                    __m256i do0_unpack_5_256;
                    __m256i do0_unpack_6_256;
                    __m256i do0_unpack_7_256;
                    quantization::Quantization8Strategy::unpack64(
                            do0_lo, do0_hi,
                            do0_unpack_0_256, do0_unpack_1_256, do0_unpack_2_256, do0_unpack_3_256,
                            do0_unpack_4_256, do0_unpack_5_256, do0_unpack_6_256, do0_unpack_7_256
                    );
                    // Combine 256 bit vectors into 512 bit vectors to speed up following operations
                    const __m512i do0_unpack_0_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_0_256), do0_unpack_1_256, 1);
                    const __m512i do0_unpack_1_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_2_256), do0_unpack_3_256, 1);
                    const __m512i do0_unpack_2_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_4_256), do0_unpack_5_256, 1);
                    const __m512i do0_unpack_3_512
                        = _mm512_inserti32x8(_mm512_castsi256_si512(do0_unpack_6_256), do0_unpack_7_256, 1);

                    const __m512 do0_f_0_512 = _mm512_cvtepi32_ps(do0_unpack_0_512);
                    const __m512 do0_f_1_512 = _mm512_cvtepi32_ps(do0_unpack_1_512);
                    const __m512 do0_f_2_512 = _mm512_cvtepi32_ps(do0_unpack_2_512);
                    const __m512 do0_f_3_512 = _mm512_cvtepi32_ps(do0_unpack_3_512);

                    // Compute dLdX updates
                    const int dLdO_group = (offset_1 + idx) / dLdO_qgroup_size;
                    const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                    const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                    const __m512 s0 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_0_512, dx0);
                    const __m512 s1 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_1_512, dx1);
                    const __m512 s2 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_2_512, dx2);
                    const __m512 s3 = _mm512_fmadd_ps(w_mul_dLdO_scale_vec, do0_f_3_512, dx3);

                    // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by its absolute value
                    // We use x0's sign to calculate the overall sign of the multiplication, and assign that sign to
                    __m512i x0_signs = _mm512_and_si512(x0, _mm512_1st_bit_on_epi8);
                    __m512i x0_abs = _mm512_abs_epi8(x0);
                    __m512i do0_abs = _mm512_abs_epi8(do0);
                    __m512i do0_signs = _mm512_and_si512(do0, _mm512_1st_bit_on_epi8);
                    __mmask64 mask = _mm512_cmp_epi8_mask(x0_signs, do0_signs, _MM_CMPINT_NE);
                    __m512i do0_signed = _mm512_mask_sub_epi8(do0_abs, mask, _mm512_zeros_epi32, do0_abs);
                    const __m512i acc_vec_i = _mm512_dpbusd_epi32(_mm512_zeros_epi32, x0_abs, do0_signed);
                    const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);

                    const int X_group = (offset_0 + idx) / X_qgroup_size;
                    const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];
                    const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                    acc_vec = _mm512_fmadd_ps(acc_vec_f, X_mul_dLdO_scale_vec, acc_vec);

                    _mm512_storeu_ps(dLdX_base, s0);
                    _mm512_storeu_ps(dLdX_base + 16, s1);
                    _mm512_storeu_ps(dLdX_base + 32, s2);
                    _mm512_storeu_ps(dLdX_base + 48, s3);
                }
                const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

                for (; idx < B; idx++) {
                    const float dx0 = dLdX_ptr[idx];
                    const int x0 = (int) X_ptr[idx];
                    const int do0 = (int) dLdO_ptr[idx];

                    const int dLdO_group = (offset_1 + idx) / dLdO_qgroup_size;
                    const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                    const int X_group = (offset_0 + idx) / X_qgroup_size;
                    const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];

                    const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                    const float dx_update = dx0 + do0_mul_w;
                    const int x0_mul_do0 = x0 * do0;

                    const float s_acc_f = X_mul_dLdO_scale * x0_mul_do0;
                    s_acc_1 += s_acc_f;
                    dLdX_ptr[idx] = dx_update;
                }

                dLdW_val[j] = s_acc_0 + s_acc_1;
            }
        }
    }
#else
    sparse::quantized_grouped_sparse_linear_vectorized_backward(
        B, M, N, X, X_scale, X_qgroup_size, W_idx_N, W_idx_M, W_val, W_scale, W_qgroup_size, dLdO, dLdO_scale, dLdO_qgroup_size, dLdX, dLdW_val
    );
#endif
}