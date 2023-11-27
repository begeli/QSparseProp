#include "qsparse_conv2d_stride_1.h"

void sparse::quantized_sparse_conv2d_vectorized_forward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    for (int oc = 0; oc < OC; oc++){
        const int oi_2 = oc * oi_0;

        for (int ic = 0; ic < IC; ic++){
            int oc_start = W_idx_OC[oc];
            int offset = ICp1 * oc + ic;
            int ic_start = oc_start + W_idx_IC[offset];
            int ic_end = oc_start + W_idx_IC[offset + 1];
            const int xi_2 = ic * xi_0;

            for (int si = ic_start; si < ic_end; si++) {
                const uint8_t i = W_idx_X[si];
                const uint8_t j = W_idx_Y[si];

                const float weight = (float) W_val[si] * W_scale;
                const float w_mul_x_scale = weight * X_scale;
                const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);

                const int pdmi = padding - i;
                const int pdmj = padding - j;
                const int p_start = std::max(pdmi, 0);
                const int p_end = std::min(pdmi + M, OM);
                const int q_start = std::max(pdmj, 0);
                const int q_end = std::min(pdmj + N, ON);
                const int px_start_tmp_0 = p_start - padding;
                const int px_start = px_start_tmp_0 + i;

                for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                    const int qx_start_tmp_0 = q_start - padding;
                    const int qx_start = qx_start_tmp_0 + j;

                    const int xi_3 = px * xi_1;
                    const int oi_3 = po * oi_1;

                    for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                        const int xi_4 = qx * B;
                        const int xi_5 = xi_2 + xi_3;
                        const int xi_6 = xi_5 + xi_4;
                        const int oi_4 = qo * B;
                        const int oi_5 = oi_2 + oi_3;
                        const int oi_6 = oi_5 + oi_4;

                        int8_t* X_ptr = X + xi_6;
                        float* O_ptr = O + oi_6;
                        int b = 0;
                        for (; b < B - 63; b += 64) {
                            float* O_base = O_ptr + b;
                            const __m512 o_0 = _mm512_loadu_ps(O_base);
                            const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                            const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                            const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                            int8_t* X_base = X_ptr + b;
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

                        for (; b < B; b++) {
                            const float o = O_ptr[b];
                            const float x = (float) X_ptr[b];

                            const float x_mul_w = w_mul_x_scale * x;
                            const float o_update = x_mul_w + o;

                            O_ptr[b] = o_update;
                        }
                    }
                }
            }
        }
    }
}

void sparse::quantized_sparse_conv2d_vectorized_parallel_forward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    #pragma omp parallel default(none) shared(B, IC, OC, M, N, K, W_nnz, padding, OM, ON, ICp1, xi_0, xi_1, oi_0, oi_1, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O)
    {
        #pragma omp for
        for (int oc = 0; oc < OC; oc++){
            const int oi_2 = oc * oi_0;

            for (int ic = 0; ic < IC; ic++){
                int oc_start = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_start = oc_start + W_idx_IC[offset];
                int ic_end = oc_start + W_idx_IC[offset + 1];
                const int xi_2 = ic * xi_0;

                for (int si = ic_start; si < ic_end; si++) {
                    const uint8_t i = W_idx_X[si];
                    const uint8_t j = W_idx_Y[si];

                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_x_scale = weight * X_scale;
                    const __m512 w_mul_x_scale_vec = _mm512_set1_ps(w_mul_x_scale);

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);
                    const int px_start_tmp_0 = p_start - padding;
                    const int px_start = px_start_tmp_0 + i;

                    for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                        const int qx_start_tmp_0 = q_start - padding;
                        const int qx_start = qx_start_tmp_0 + j;

                        const int xi_3 = px * xi_1;
                        const int oi_3 = po * oi_1;

                        for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                            const int xi_4 = qx * B;
                            const int xi_5 = xi_2 + xi_3;
                            const int xi_6 = xi_5 + xi_4;
                            const int oi_4 = qo * B;
                            const int oi_5 = oi_2 + oi_3;
                            const int oi_6 = oi_5 + oi_4;

                            int8_t* X_ptr = X + xi_6;
                            float* O_ptr = O + oi_6;
                            int b = 0;
                            for (; b < B - 63; b += 64) {
                                float* O_base = O_ptr + b;
                                const __m512 o_0 = _mm512_loadu_ps(O_base);
                                const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                                const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                                const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                                int8_t* X_base = X_ptr + b;
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

                            for (; b < B; b++) {
                                const float o = O_ptr[b];
                                const float x = (float) X_ptr[b];

                                const float x_mul_w = w_mul_x_scale * x;
                                const float o_update = x_mul_w + o;

                                O_ptr[b] = o_update;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    sparse::quantized_sparse_conv2d_vectorized_forward_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O
    );
#endif
}

void sparse::quantized_grouped_sparse_conv2d_vectorized_forward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    for (int oc = 0; oc < OC; oc++){
        const int oi_2 = oc * oi_0;

        for (int ic = 0; ic < IC; ic++){
            int oc_start = W_idx_OC[oc];
            int offset = ICp1 * oc + ic;
            int ic_start = oc_start + W_idx_IC[offset];
            int ic_end = oc_start + W_idx_IC[offset + 1];
            const int xi_2 = ic * xi_0;

            for (int si = ic_start; si < ic_end; si++) {
                const uint8_t i = W_idx_X[si];
                const uint8_t j = W_idx_Y[si];

                const int W_group = si / W_qgroup_size;
                const float weight = (float) W_val[si] * W_scale[W_group];

                const int pdmi = padding - i;
                const int pdmj = padding - j;
                const int p_start = std::max(pdmi, 0);
                const int p_end = std::min(pdmi + M, OM);
                const int q_start = std::max(pdmj, 0);
                const int q_end = std::min(pdmj + N, ON);
                const int px_start_tmp_0 = p_start - padding;
                const int px_start = px_start_tmp_0 + i;

                for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                    const int qx_start_tmp_0 = q_start - padding;
                    const int qx_start = qx_start_tmp_0 + j;

                    const int xi_3 = px * xi_1;
                    const int oi_3 = po * oi_1;

                    for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                        const int xi_4 = qx * B;
                        const int xi_5 = xi_2 + xi_3;
                        const int xi_6 = xi_5 + xi_4;
                        const int oi_4 = qo * B;
                        const int oi_5 = oi_2 + oi_3;
                        const int oi_6 = oi_5 + oi_4;

                        int8_t* X_ptr = X + xi_6;
                        float* O_ptr = O + oi_6;
                        int b = 0;
                        for (; b < B - 63; b += 64) {
                            float* O_base = O_ptr + b;
                            const __m512 o_0 = _mm512_loadu_ps(O_base);
                            const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                            const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                            const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                            int8_t* X_base = X_ptr + b;
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

                            const int X_group = (xi_6 + b) / X_qgroup_size;
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

                        for (; b < B; b++) {
                            const float o = O_ptr[b];
                            const float x = (float) X_ptr[b];

                            const int X_group = (xi_6 + b) / X_qgroup_size;
                            const float w_mul_x_scale = weight * X_scale[X_group];

                            const float x_mul_w = w_mul_x_scale * x;
                            const float o_update = x_mul_w + o;

                            O_ptr[b] = o_update;
                        }
                    }
                }
            }
        }
    }
}

void sparse::quantized_grouped_sparse_conv2d_vectorized_parallel_forward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size, float* O
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    #pragma omp parallel default(none) shared(B, IC, OC, M, N, K, W_nnz, padding, OM, ON, ICp1, xi_0, xi_1, oi_0, oi_1, X, X_scale, X_qgroup_size, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, W_qgroup_size, O)
    {
        #pragma omp for
        for (int oc = 0; oc < OC; oc++) {
            const int oi_2 = oc * oi_0;

            for (int ic = 0; ic < IC; ic++) {
                int oc_start = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_start = oc_start + W_idx_IC[offset];
                int ic_end = oc_start + W_idx_IC[offset + 1];
                const int xi_2 = ic * xi_0;

                for (int si = ic_start; si < ic_end; si++) {
                    const uint8_t i = W_idx_X[si];
                    const uint8_t j = W_idx_Y[si];

                    const int W_group = si / W_qgroup_size;
                    const float weight = (float) W_val[si] * W_scale[W_group];

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);
                    const int px_start_tmp_0 = p_start - padding;
                    const int px_start = px_start_tmp_0 + i;

                    for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                        const int qx_start_tmp_0 = q_start - padding;
                        const int qx_start = qx_start_tmp_0 + j;

                        const int xi_3 = px * xi_1;
                        const int oi_3 = po * oi_1;

                        for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                            const int xi_4 = qx * B;
                            const int xi_5 = xi_2 + xi_3;
                            const int xi_6 = xi_5 + xi_4;
                            const int oi_4 = qo * B;
                            const int oi_5 = oi_2 + oi_3;
                            const int oi_6 = oi_5 + oi_4;

                            int8_t *X_ptr = X + xi_6;
                            float *O_ptr = O + oi_6;
                            int b = 0;
                            for (; b < B - 63; b += 64) {
                                float *O_base = O_ptr + b;
                                const __m512 o_0 = _mm512_loadu_ps(O_base);
                                const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                                const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                                const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                                int8_t *X_base = X_ptr + b;
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

                                const int X_group = (xi_6 + b) / X_qgroup_size;
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

                            for (; b < B; b++) {
                                const float o = O_ptr[b];
                                const float x = (float) X_ptr[b];

                                const int X_group = (xi_6 + b) / X_qgroup_size;
                                const float w_mul_x_scale = weight * X_scale[X_group];

                                const float x_mul_w = w_mul_x_scale * x;
                                const float o_update = x_mul_w + o;

                                O_ptr[b] = o_update;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    sparse::quantized_grouped_sparse_conv2d_vectorized_forward_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, X_qgroup_size, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, W_qgroup_size, O
    );
#endif
}

void sparse::quantized_sparse_conv2d_vectorized_backward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale,
    int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    for (int ic = 0; ic < IC; ic++) {
        const int xi_2 = ic * xi_0;

        for (int oc = 0; oc < OC; oc++) {
            int oc_start = W_idx_OC[oc];
            int offset = ICp1 * oc + ic;
            int ic_start = oc_start + W_idx_IC[offset];
            int ic_end = oc_start + W_idx_IC[offset + 1];
            const int oi_2 = oc * oi_0;

            for (int si = ic_start; si < ic_end; si++) {
                uint8_t i = W_idx_X[si];
                uint8_t j = W_idx_Y[si];

                const float weight = (float) W_val[si] * W_scale;
                const float w_mul_dLdO_scale = weight * dLdO_scale;
                const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                int s_acc_i = 0;
                __m512i acc_vec_i = _mm512_setzero_epi32();

                const int pdmi = padding - i;
                const int pdmj = padding - j;
                const int p_start = std::max(pdmi, 0);
                const int p_end = std::min(pdmi + M, OM);
                const int q_start = std::max(pdmj, 0);
                const int q_end = std::min(pdmj + N, ON);
                const int px_start_tmp_0 = p_start - padding;
                const int px_start = px_start_tmp_0 + i;

                for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                    const int qx_start_tmp_0 = q_start - padding;
                    const int qx_start = qx_start_tmp_0 + j;
                    const int xi_3 = px * xi_1;
                    const int oi_3 = po * oi_1;

                    for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                        const int xi_4 = qx * B;
                        const int xi_5 = xi_2 + xi_3;
                        const int xi_6 = xi_5 + xi_4;
                        const int oi_4 = qo * B;
                        const int oi_5 = oi_2 + oi_3;
                        const int oi_6 = oi_5 + oi_4;

                        int8_t* X_ptr = X + xi_6;
                        int8_t*  dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_6;
                        int b = 0;
                        for (; b < B - 63; b += 64) {
                            const __m512i x0 = _mm512_loadu_epi32(X_ptr + b);
                            const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + b);

                            float* dLdX_base = dLdX_ptr + b;
                            const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                            const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                            const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                            const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);

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

                        for (; b < B; b++) {
                            const int x0 = (int) X_ptr[b];
                            const int do0 = (int) dLdO_ptr[b];
                            const float dx0 = dLdX_ptr[b];

                            const int x0_mul_do0 = x0 * do0;
                            const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                            const float dx_update = dx0 + do0_mul_w;

                            s_acc_i = s_acc_i + x0_mul_do0;
                            dLdX_ptr[b] = dx_update;
                        }
                    }
                }
                const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                const __m512 acc_vec = _mm512_mul_ps(X_mul_dLdO_scale_vec, acc_vec_f);
                const float s_acc_0 = _mm512_haddf32_ss(acc_vec);
                const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

                dLdW_val[si] += s_acc_0 + s_acc_1;
            }
        }
    }
}

void sparse::quantized_sparse_conv2d_vectorized_parallel_backward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale,
    int8_t * dLdO, float dLdO_scale, float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    #pragma omp parallel default(none) shared(B, IC, OC, M, N, K, W_nnz, padding, OM, ON, ICp1, xi_0, xi_1, oi_0, oi_1, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, dLdO, dLdO_scale, dLdX, dLdW_val, _mm512_1st_bit_on_epi8, _mm512_zeros_epi32)
    {
        #pragma omp for reduction(+:dLdW_val[:W_nnz])
        for (int ic = 0; ic < IC; ic++) {
            const int xi_2 = ic * xi_0;

            for (int oc = 0; oc < OC; oc++) {
                int oc_start = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_start = oc_start + W_idx_IC[offset];
                int ic_end = oc_start + W_idx_IC[offset + 1];
                const int oi_2 = oc * oi_0;

                for (int si = ic_start; si < ic_end; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_dLdO_scale = weight * dLdO_scale;
                    const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                    const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                    const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                    int s_acc_i = 0;
                    __m512i acc_vec_i = _mm512_setzero_epi32();

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);
                    const int px_start_tmp_0 = p_start - padding;
                    const int px_start = px_start_tmp_0 + i;

                    for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                        const int qx_start_tmp_0 = q_start - padding;
                        const int qx_start = qx_start_tmp_0 + j;
                        const int xi_3 = px * xi_1;
                        const int oi_3 = po * oi_1;

                        for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                            const int xi_4 = qx * B;
                            const int xi_5 = xi_2 + xi_3;
                            const int xi_6 = xi_5 + xi_4;
                            const int oi_4 = qo * B;
                            const int oi_5 = oi_2 + oi_3;
                            const int oi_6 = oi_5 + oi_4;

                            int8_t* X_ptr = X + xi_6;
                            int8_t*  dLdO_ptr = dLdO + oi_6;
                            float* dLdX_ptr = dLdX + xi_6;
                            int b = 0;
                            for (; b < B - 63; b += 64) {
                                const __m512i x0 = _mm512_loadu_epi32(X_ptr + b);
                                const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + b);

                                float* dLdX_base = dLdX_ptr + b;
                                const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                                const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                                const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                                const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);

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

                            for (; b < B; b++) {
                                const int x0 = (int) X_ptr[b];
                                const int do0 = (int) dLdO_ptr[b];
                                const float dx0 = dLdX_ptr[b];

                                const int x0_mul_do0 = x0 * do0;
                                const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                                const float dx_update = dx0 + do0_mul_w;

                                s_acc_i = s_acc_i + x0_mul_do0;
                                dLdX_ptr[b] = dx_update;
                            }
                        }
                    }
                    const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                    const __m512 acc_vec = _mm512_mul_ps(X_mul_dLdO_scale_vec, acc_vec_f);
                    const float s_acc_0 = _mm512_haddf32_ss(acc_vec);
                    const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

                    dLdW_val[si] += s_acc_0 + s_acc_1;
                }
            }
        }
    }
#else
    sparse::quantized_sparse_conv2d_vectorized_backward_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X,
        W_idx_Y, W_val, W_scale, dLdO, dLdO_scale, dLdX, dLdW_val
    );
#endif
}

void sparse::quantized_grouped_sparse_conv2d_vectorized_backward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size,
    int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    for (int ic = 0; ic < IC; ic++) {
        const int xi_2 = ic * xi_0;

        for (int oc = 0; oc < OC; oc++) {
            int oc_start = W_idx_OC[oc];
            int offset = ICp1 * oc + ic;
            int ic_start = oc_start + W_idx_IC[offset];
            int ic_end = oc_start + W_idx_IC[offset + 1];
            const int oi_2 = oc * oi_0;

            for (int si = ic_start; si < ic_end; si++) {
                uint8_t i = W_idx_X[si];
                uint8_t j = W_idx_Y[si];

                const int W_group = si / W_qgroup_size;
                const float weight = (float) W_val[si] * W_scale[W_group];
                float s_acc_1 = 0.0;
                __m512 acc_vec = _mm512_setzero_ps();

                const int pdmi = padding - i;
                const int pdmj = padding - j;
                const int p_start = std::max(pdmi, 0);
                const int p_end = std::min(pdmi + M, OM);
                const int q_start = std::max(pdmj, 0);
                const int q_end = std::min(pdmj + N, ON);
                const int px_start_tmp_0 = p_start - padding;
                const int px_start = px_start_tmp_0 + i;

                for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                    const int qx_start_tmp_0 = q_start - padding;
                    const int qx_start = qx_start_tmp_0 + j;
                    const int xi_3 = px * xi_1;
                    const int oi_3 = po * oi_1;

                    for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                        const int xi_4 = qx * B;
                        const int xi_5 = xi_2 + xi_3;
                        const int xi_6 = xi_5 + xi_4;
                        const int oi_4 = qo * B;
                        const int oi_5 = oi_2 + oi_3;
                        const int oi_6 = oi_5 + oi_4;

                        int8_t* X_ptr = X + xi_6;
                        int8_t*  dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_6;
                        int b = 0;
                        for (; b < B - 63; b += 64) {
                            const __m512i x0 = _mm512_loadu_epi32(X_ptr + b);
                            const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + b);

                            float* dLdX_base = dLdX_ptr + b;
                            const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                            const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                            const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                            const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);

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
                            const int dLdO_group = (oi_6 + b) / dLdO_qgroup_size;
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

                            const int X_group = (xi_6 + b) / X_qgroup_size;
                            const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];
                            const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                            const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                            acc_vec = _mm512_fmadd_ps(X_mul_dLdO_scale_vec, acc_vec_f, acc_vec);

                            _mm512_storeu_ps(dLdX_base, s0);
                            _mm512_storeu_ps(dLdX_base + 16, s1);
                            _mm512_storeu_ps(dLdX_base + 32, s2);
                            _mm512_storeu_ps(dLdX_base + 48, s3);
                        }

                        for (; b < B; b++) {
                            const int x0 = (int) X_ptr[b];
                            const int do0 = (int) dLdO_ptr[b];
                            const float dx0 = dLdX_ptr[b];

                            const int dLdO_group = (oi_6 + b) / dLdO_qgroup_size;
                            const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                            const int X_group = (xi_6 + b) / X_qgroup_size;
                            const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];

                            const int x0_mul_do0 = x0 * do0;
                            const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                            const float dx_update = dx0 + do0_mul_w;

                            const float s_acc_f = X_mul_dLdO_scale * x0_mul_do0;
                            s_acc_1 += s_acc_f;
                            dLdX_ptr[b] = dx_update;
                        }
                    }
                }
                const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

                dLdW_val[si] += s_acc_0 + s_acc_1;
            }
        }
    }
}

void sparse::quantized_grouped_sparse_conv2d_vectorized_parallel_backward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float* X_scale, int X_qgroup_size, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float* W_scale, int W_qgroup_size,
    int8_t * dLdO, float* dLdO_scale, int dLdO_qgroup_size, float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int xi_0 = B * M * N;
    const int xi_1 = N * B;
    const int oi_0 = OM * ON * B;
    const int oi_1 = ON * B;

    #pragma omp parallel default(none) shared(B, IC, OC, M, N, K, W_nnz, padding, OM, ON, ICp1, xi_0, xi_1, oi_0, oi_1, X, X_scale, X_qgroup_size, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, W_qgroup_size, dLdO, dLdO_scale, dLdO_qgroup_size, dLdX, dLdW_val, _mm512_1st_bit_on_epi8, _mm512_zeros_epi32)
    {
        #pragma omp for reduction(+:dLdW_val[:W_nnz])
        for (int ic = 0; ic < IC; ic++) {
            const int xi_2 = ic * xi_0;

            for (int oc = 0; oc < OC; oc++) {
                int oc_start = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_start = oc_start + W_idx_IC[offset];
                int ic_end = oc_start + W_idx_IC[offset + 1];
                const int oi_2 = oc * oi_0;

                for (int si = ic_start; si < ic_end; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    const int W_group = si / W_qgroup_size;
                    const float weight = (float) W_val[si] * W_scale[W_group];
                    float s_acc_1 = 0.0;
                    __m512 acc_vec = _mm512_setzero_ps();

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);
                    const int px_start_tmp_0 = p_start - padding;
                    const int px_start = px_start_tmp_0 + i;

                    for (int po = p_start, px = px_start; po < p_end; po++, px++) {
                        const int qx_start_tmp_0 = q_start - padding;
                        const int qx_start = qx_start_tmp_0 + j;
                        const int xi_3 = px * xi_1;
                        const int oi_3 = po * oi_1;

                        for (int qo = q_start, qx = qx_start; qo < q_end; qo++, qx++) {
                            const int xi_4 = qx * B;
                            const int xi_5 = xi_2 + xi_3;
                            const int xi_6 = xi_5 + xi_4;
                            const int oi_4 = qo * B;
                            const int oi_5 = oi_2 + oi_3;
                            const int oi_6 = oi_5 + oi_4;

                            int8_t *X_ptr = X + xi_6;
                            int8_t *dLdO_ptr = dLdO + oi_6;
                            float *dLdX_ptr = dLdX + xi_6;
                            int b = 0;
                            for (; b < B - 63; b += 64) {
                                const __m512i x0 = _mm512_loadu_epi32(X_ptr + b);
                                const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + b);

                                float *dLdX_base = dLdX_ptr + b;
                                const __m512 dx0 = _mm512_loadu_ps(dLdX_base);
                                const __m512 dx1 = _mm512_loadu_ps(dLdX_base + 16);
                                const __m512 dx2 = _mm512_loadu_ps(dLdX_base + 32);
                                const __m512 dx3 = _mm512_loadu_ps(dLdX_base + 48);

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
                                const int dLdO_group = (oi_6 + b) / dLdO_qgroup_size;
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

                                const int X_group = (xi_6 + b) / X_qgroup_size;
                                const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];
                                const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                                const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                                acc_vec = _mm512_fmadd_ps(X_mul_dLdO_scale_vec, acc_vec_f, acc_vec);

                                _mm512_storeu_ps(dLdX_base, s0);
                                _mm512_storeu_ps(dLdX_base + 16, s1);
                                _mm512_storeu_ps(dLdX_base + 32, s2);
                                _mm512_storeu_ps(dLdX_base + 48, s3);
                            }

                            for (; b < B; b++) {
                                const int x0 = (int) X_ptr[b];
                                const int do0 = (int) dLdO_ptr[b];
                                const float dx0 = dLdX_ptr[b];

                                const int dLdO_group = (oi_6 + b) / dLdO_qgroup_size;
                                const float w_mul_dLdO_scale = weight * dLdO_scale[dLdO_group];
                                const int X_group = (xi_6 + b) / X_qgroup_size;
                                const float X_mul_dLdO_scale = X_scale[X_group] * dLdO_scale[dLdO_group];

                                const int x0_mul_do0 = x0 * do0;
                                const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                                const float dx_update = dx0 + do0_mul_w;

                                const float s_acc_f = X_mul_dLdO_scale * x0_mul_do0;
                                s_acc_1 += s_acc_f;
                                dLdX_ptr[b] = dx_update;
                            }
                        }
                    }
                    const float s_acc_0 = _mm512_haddf32_ss(acc_vec);

                    dLdW_val[si] += s_acc_0 + s_acc_1;
                }
            }
        }
    }
#else
    sparse::quantized_grouped_sparse_conv2d_vectorized_backward_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, X_qgroup_size, W_idx_OC, W_idx_IC, W_idx_X,
        W_idx_Y, W_val, W_scale, W_qgroup_size, dLdO, dLdO_scale, dLdO_qgroup_size, dLdX, dLdW_val
    );
#endif
}