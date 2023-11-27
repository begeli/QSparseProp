#include "qsparse_conv2d_over_on.h"

#include <iostream>

void sparse::quantized_sparse_conv2d_vectorized_forward_over_on_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int oc = 0; oc < OC; oc++) {
            const int oc_s = W_idx_OC[oc];
            const int oi_3 = oc * oi_1;
            const int oi_4 = oi_2 + oi_3;
            for (int ic = 0; ic < IC; ic++) {
                int offset = ICp1 * oc + ic;
                int ic_s = oc_s + W_idx_IC[offset];
                int ic_e = oc_s + W_idx_IC[offset + 1];
                const int xi_3 = ic * xi_1;
                const int xi_4 = xi_2 + xi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_x_scale = weight * X_scale;
                    const __m256 w_mul_x_scale_vec = _mm256_set1_ps(w_mul_x_scale);

                    int pdmi = padding - i;
                    int pdmj = padding - j;
                    int p_start = std::max(pdmi, 0);
                    int p_end = std::min(pdmi + M, OM);
                    int q_start = std::max(pdmj, 0);
                    int q_end = std::min(pdmj + N, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;

                        float* O_ptr = O + oi_6;
                        int8_t* X_ptr = X + xi_6;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            float* O_base = O_ptr + q;
                            int8_t* X_base = X_ptr + (q - pdmj);

                            __m256 ov = _mm256_loadu_ps(O_base);
                            __m128i xv_packed_i = _mm_loadu_si64(X_base);
                            __m256i xv_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(xv_packed_i, xv_unpacked_i);

                            __m256 xv_f = _mm256_cvtepi32_ps(xv_unpacked_i);
                            ov = _mm256_fmadd_ps(w_mul_x_scale_vec, xv_f, ov);

                            _mm256_storeu_ps(O_base, ov);
                            /*float* O_base = O_ptr + q;
                            const __m512 o_0 = _mm512_loadu_ps(O_base);
                            const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                            const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                            const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                            int8_t* X_base = X_ptr + (q - pdmj);
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
                            _mm512_storeu_ps(O_base + 48, r3);*/
                        }

                        for (; q < q_end; q++) {
                            const float o = O_ptr[q];
                            const float x = (float) X_ptr[q - pdmj];

                            const float x_mul_w = w_mul_x_scale * x;
                            const float o_update = x_mul_w + o;

                            O_ptr[q] = o_update;
                        }
                    }
                }
            }
        }
    }
}

void sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
    float* dLdX, float* dLdW_val
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int ic = 0; ic < IC; ic++){
            const int xi_3 = ic * xi_1;
            const int xi_4 = xi_2 + xi_3;
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_s = oc_s + W_idx_IC[offset];
                int ic_e = oc_s + W_idx_IC[offset + 1];
                const int oi_3 = oc * oi_1;
                const int oi_4 = oi_2 + oi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_dLdO_scale = weight * dLdO_scale;
                    const __m256 w_mul_dLdO_scale_vec = _mm256_set1_ps(w_mul_dLdO_scale);
                    const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                    const __m256 X_mul_dLdO_scale_vec = _mm256_set1_ps(X_mul_dLdO_scale);
                    int s_acc_i = 0;
                    __m256i acc_vec_i = _mm256_set1_ps(0.0f);

                    int pdmi = padding - i;
                    int pdmj = padding - j;
                    int p_start = std::max(pdmi, 0);
                    int p_end = std::min(pdmi + M, OM);
                    int q_start = std::max(pdmj, 0);
                    int q_end = std::min(pdmj + N, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;
                        const int xi_7 = xi_6 - pdmj;

                        int8_t* X_ptr = X + xi_7;
                        int8_t*  dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_7;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            const __m128i ov_packed_i = _mm_loadu_si64(X_ptr + q);
                            const __m128i xv_packed_i = _mm_loadu_si64(dLdO_ptr + q);
                            const __m256 dxv = _mm256_loadu_ps(dLdX_ptr + q);

                            __m256i ov_unpacked_i;
                            __m256i xv_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(ov_packed_i, ov_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(xv_packed_i, xv_unpacked_i);

                            const __m256 ov_f = _mm256_cvtepi32_ps(ov_unpacked_i);
                            const __m256 xv_f = _mm256_cvtepi32_ps(xv_unpacked_i);

                            acc_vec_i = _mm256_fmadd_ps(ov_f, xv_f, acc_vec_i);

                            const __m256 dxv_new = _mm256_fmadd_ps(w_mul_dLdO_scale_vec, ov_f, dxv);
                            _mm256_storeu_ps(dLdX_ptr + q, dxv_new);
                            /*const __m512i x0 = _mm512_loadu_epi32(X_ptr + q);
                            const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + q);

                            float* dLdX_base = dLdX_ptr + q;
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

                            // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by taking its absolute value
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
                            _mm512_storeu_ps(dLdX_base + 48, s3);*/
                        }

                        for (; q < q_end; q++) {
                            const int x0 = (int) X_ptr[q];
                            const int do0 = (int) dLdO_ptr[q];
                            const float dx0 = dLdX_ptr[q];

                            const int x0_mul_do0 = x0 * do0;
                            const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                            const float dx_update = dx0 + do0_mul_w;

                            s_acc_i = s_acc_i + x0_mul_do0;
                            dLdX_ptr[q] = dx_update;
                        }
                    }
                    const __m256 acc_vec = _mm256_mul_ps(X_mul_dLdO_scale_vec, acc_vec_i);
                    const float s_acc_0 = _mm256_haddf32_ss(acc_vec);
                    const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

                    dLdW_val[si] += s_acc_0 + s_acc_1;
                }
            }
        }
    }
}

void sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_2(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
    float* dLdX, float* dLdW_val
) {
    int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    __m256 x_scale_v = _mm256_set1_ps(X_scale);
    __m256 dLdO_scale_v = _mm256_set1_ps(dLdO_scale);
    __m256i permutevar8x32_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    __m256 zv = _mm256_setzero_ps();
    __m256i permuteamask = _mm256_set_epi32(7,7,7,7,6,4,2,0);
    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int ic = 0; ic < IC; ic++) {
            const int xi_3 = ic * xi_1;
            const int xi_4 = xi_2 + xi_3;
            for (int oc = 0; oc < OC; oc++) {
                const int oc_s = W_idx_OC[oc];
                const int offset = ICp1 * oc + ic;
                const int ic_s = oc_s + W_idx_IC[offset];
                const int ic_e = oc_s + W_idx_IC[offset + 1];
                const int oi_3 = oc * oi_1;
                const int oi_4 = oi_2 + oi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    const uint8_t i = W_idx_X[si];
                    const uint8_t j = W_idx_Y[si];

                    //float x_scale_mul_dLdO_scale = X_scale * dLdO_scale;
                    const float v = (float) W_val[si] * W_scale;
                    const __m256 v0v = _mm256_set_ps(0., v, 0., v, 0., v, 0., v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (2 * p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;
                        const int xi_7 = xi_6 - pdmj;

                        int8_t* X_ptr = X + xi_7;
                        int8_t* dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_7;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            const __m128i ov_packed_i = _mm_loadu_si64(dLdO_ptr + q);
                            const __m128i x_0_packed_i = _mm_loadu_si64(X_ptr + 2 * q);
                            const __m128i x_1_packed_i = _mm_loadu_si64(X_ptr + 2 * q + 8);

                            // Unpack
                            __m256i ov_unpacked_i;
                            __m256i x_0_unpacked_i;
                            __m256i x_1_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(ov_packed_i, ov_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(x_0_packed_i, x_0_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(x_1_packed_i, x_1_unpacked_i);

                            // Convert from epi32 to ps
                            const __m256 ov_unpacked_f = _mm256_cvtepi32_ps(ov_unpacked_i);
                            const __m256 x_0_unpacked_f = _mm256_cvtepi32_ps(x_0_unpacked_i);
                            const __m256 x_1_unpacked_f = _mm256_cvtepi32_ps(x_1_unpacked_i);

                            // Dequantize
                            const __m256 ov_dq_f = _mm256_mul_ps(ov_unpacked_f, dLdO_scale_v);

                            // Permute for stride
                            const __m256 x_0_p = _mm256_permutevar8x32_ps(x_0_unpacked_f, permuteamask);
                            const __m256 x_1_p = _mm256_permutevar8x32_ps(x_1_unpacked_f, permuteamask);
                            const __m256 xv  = _mm256_insertf128_ps(x_0_p, _mm256_castps256_ps128(x_1_p), 1);
                            const __m256 xv_dq = _mm256_mul_ps(xv, x_scale_v);

                            dwv = _mm256_fmadd_ps(ov_dq_f, xv_dq, dwv);

                            const __m256 ov = _mm256_permutevar8x32_ps(ov_dq_f, permutevar8x32_mask);
                            const __m256 ov0 = _mm256_unpacklo_ps(ov, zv);
                            const __m256 ov1 = _mm256_unpackhi_ps(ov, zv);

                            const __m256 dxv0 = _mm256_loadu_ps(dLdX_ptr + 2 * q);
                            const __m256 dxv1 = _mm256_loadu_ps(dLdX_ptr + 2 * q + 8);

                            const __m256 dxv0_new = _mm256_fmadd_ps(ov0, v0v, dxv0);
                            const __m256 dxv1_new = _mm256_fmadd_ps(ov1, v0v, dxv1);

                            _mm256_storeu_ps(dLdX_ptr + 2 * q, dxv0_new);
                            _mm256_storeu_ps(dLdX_ptr + 2 * q + 8, dxv1_new);
                        }

                        for (; q < q_end; q++) {
                            float o = (float) dLdO_ptr[q] * dLdO_scale;
                            float x = (float) X_ptr[2 * q] * X_scale;

                            dLdX_ptr[2 * q] += v * o;
                            dw += o * x;
                        }
                    }
                    dw += _mm256_haddf32_ss(dwv);
                    dLdW_val[si] += dw;
                }
            }
        }
    }
}

void sparse::quantized_sparse_conv2d_vectorized_parallel_forward_over_on_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, float* O
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    #pragma omp parallel for collapse(2) default(none) shared(oi_0, oi_1, xi_0, xi_1, B, OC, W_idx_OC, IC, ICp1, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, X_scale, padding, M, OM, N, ON, X, O)
    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int oc = 0; oc < OC; oc++) {
            const int oc_s = W_idx_OC[oc];
            const int oi_3 = oc * oi_1;
            const int oi_4 = oi_2 + oi_3;
            for (int ic = 0; ic < IC; ic++) {
                int offset = ICp1 * oc + ic;
                int ic_s = oc_s + W_idx_IC[offset];
                int ic_e = oc_s + W_idx_IC[offset + 1];
                const int xi_3 = ic * xi_1;
                const int xi_4 = xi_2 + xi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_x_scale = weight * X_scale;
                    const __m256 w_mul_x_scale_vec = _mm256_set1_ps(w_mul_x_scale);

                    int pdmi = padding - i;
                    int pdmj = padding - j;
                    int p_start = std::max(pdmi, 0);
                    int p_end = std::min(pdmi + M, OM);
                    int q_start = std::max(pdmj, 0);
                    int q_end = std::min(pdmj + N, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;

                        float* O_ptr = O + oi_6;
                        int8_t* X_ptr = X + xi_6;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            float* O_base = O_ptr + q;
                            int8_t* X_base = X_ptr + (q - pdmj);

                            __m256 ov = _mm256_loadu_ps(O_base);
                            __m128i xv_packed_i = _mm_loadu_si64(X_base);
                            __m256i xv_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(xv_packed_i, xv_unpacked_i);

                            __m256 xv_f = _mm256_cvtepi32_ps(xv_unpacked_i);
                            ov = _mm256_fmadd_ps(w_mul_x_scale_vec, xv_f, ov);

                            _mm256_storeu_ps(O_base, ov);
                            /*float* O_base = O_ptr + q;
                            const __m512 o_0 = _mm512_loadu_ps(O_base);
                            const __m512 o_1 = _mm512_loadu_ps(O_base + 16);
                            const __m512 o_2 = _mm512_loadu_ps(O_base + 32);
                            const __m512 o_3 = _mm512_loadu_ps(O_base + 48);

                            int8_t* X_base = X_ptr + (q - pdmj);
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
                            _mm512_storeu_ps(O_base + 48, r3);*/
                        }

                        for (; q < q_end; q++) {
                            const float o = O_ptr[q];
                            const float x = (float) X_ptr[q - pdmj];

                            const float x_mul_w = w_mul_x_scale * x;
                            const float o_update = x_mul_w + o;

                            O_ptr[q] = o_update;
                        }
                    }
                }
            }
        }
    }
#else
    sparse::quantized_sparse_conv2d_vectorized_forward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X,
        W_idx_Y, W_val, W_scale, O
    )
#endif
}

void sparse::quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
    float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    #pragma omp parallel for collapse(2) default(none) shared(B, OC, oi_0, xi_0, IC, xi_1, W_idx_OC, ICp1, W_idx_IC, oi_1, W_idx_X, W_idx_Y, W_val, W_scale, dLdO_scale, X_scale, padding, M, OM, N, ON, X, dLdO, dLdX, _mm512_1st_bit_on_epi8, dLdW_val, _mm512_zeros_epi32)
    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int ic = 0; ic < IC; ic++){
            const int xi_3 = ic * xi_1;
            const int xi_4 = xi_2 + xi_3;
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int offset = ICp1 * oc + ic;
                int ic_s = oc_s + W_idx_IC[offset];
                int ic_e = oc_s + W_idx_IC[offset + 1];
                const int oi_3 = oc * oi_1;
                const int oi_4 = oi_2 + oi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    /*const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_dLdO_scale = weight * dLdO_scale;
                    const __m512 w_mul_dLdO_scale_vec = _mm512_set1_ps(w_mul_dLdO_scale);
                    const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                    const __m512 X_mul_dLdO_scale_vec = _mm512_set1_ps(X_mul_dLdO_scale);
                    int s_acc_i = 0;
                    __m512i acc_vec_i = _mm512_setzero_epi32();*/
                    const float weight = (float) W_val[si] * W_scale;
                    const float w_mul_dLdO_scale = weight * dLdO_scale;
                    const __m256 w_mul_dLdO_scale_vec = _mm256_set1_ps(w_mul_dLdO_scale);
                    const float X_mul_dLdO_scale = X_scale * dLdO_scale;
                    const __m256 X_mul_dLdO_scale_vec = _mm256_set1_ps(X_mul_dLdO_scale);
                    int s_acc_i = 0;
                    __m256i acc_vec_i = _mm256_set1_ps(0.0f);

                    int pdmi = padding - i;
                    int pdmj = padding - j;
                    int p_start = std::max(pdmi, 0);
                    int p_end = std::min(pdmi + M, OM);
                    int q_start = std::max(pdmj, 0);
                    int q_end = std::min(pdmj + N, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;
                        const int xi_7 = xi_6 - pdmj;

                        int8_t* X_ptr = X + xi_7;
                        int8_t*  dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_7;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            const __m128i ov_packed_i = _mm_loadu_si64(X_ptr + q);
                            const __m128i xv_packed_i = _mm_loadu_si64(dLdO_ptr + q);
                            const __m256 dxv = _mm256_loadu_ps(dLdX_ptr + q);

                            __m256i ov_unpacked_i;
                            __m256i xv_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(ov_packed_i, ov_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(xv_packed_i, xv_unpacked_i);

                            const __m256 ov_f = _mm256_cvtepi32_ps(ov_unpacked_i);
                            const __m256 xv_f = _mm256_cvtepi32_ps(xv_unpacked_i);

                            acc_vec_i = _mm256_fmadd_ps(ov_f, xv_f, acc_vec_i);

                            const __m256 dxv_new = _mm256_fmadd_ps(w_mul_dLdO_scale_vec, ov_f, dxv);
                            _mm256_storeu_ps(dLdX_ptr + q, dxv_new);
                            /*const __m512i x0 = _mm512_loadu_epi32(X_ptr + q);
                            const __m512i do0 = _mm512_loadu_epi32(dLdO_ptr + q);

                            float* dLdX_base = dLdX_ptr + q;
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

                            // Because Intel is weird, one of the numbers must be unsigned, we make x0 unsigned by taking its absolute value
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
                            _mm512_storeu_ps(dLdX_base + 48, s3);*/
                        }

                        for (; q < q_end; q++) {
                            const int x0 = (int) X_ptr[q];
                            const int do0 = (int) dLdO_ptr[q];
                            const float dx0 = dLdX_ptr[q];

                            const int x0_mul_do0 = x0 * do0;
                            const float do0_mul_w = w_mul_dLdO_scale * (float) do0;
                            const float dx_update = dx0 + do0_mul_w;

                            s_acc_i = s_acc_i + x0_mul_do0;
                            dLdX_ptr[q] = dx_update;
                        }
                    }
                    /*const __m512 acc_vec_f = _mm512_cvtepi32_ps(acc_vec_i);
                    const __m512 acc_vec = _mm512_mul_ps(X_mul_dLdO_scale_vec, acc_vec_f);
                    const float s_acc_0 = _mm512_haddf32_ss(acc_vec);
                    const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;*/
                    const __m256 acc_vec = _mm256_mul_ps(X_mul_dLdO_scale_vec, acc_vec_i);
                    const float s_acc_0 = _mm256_haddf32_ss(acc_vec);
                    const float s_acc_1 = X_mul_dLdO_scale * (float) s_acc_i;

                    #pragma omp atomic
                    dLdW_val[si] += s_acc_0 + s_acc_1;
                }
            }
        }
    }
#else
    sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_1(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X,
        W_idx_Y, W_val, W_scale, dLdO, dLdO_scale, dLdX, dLdW_val
    );
#endif
}

void sparse::quantized_sparse_conv2d_vectorized_parallel_backward_over_on_stride_2(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    int8_t* X, float X_scale, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, int8_t* W_val, float W_scale, int8_t * dLdO, float dLdO_scale,
    float* dLdX, float* dLdW_val
) {
#if defined(_OPENMP)
    int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);
    const int ICp1 = IC + 1;
    const int oi_0 = OC * OM * ON;
    const int oi_1 = OM * ON;
    const int xi_0 = IC * M * N;
    const int xi_1 = M * N;

    __m256 x_scale_v = _mm256_set1_ps(X_scale);
    __m256 dLdO_scale_v = _mm256_set1_ps(dLdO_scale);
    __m256i permutevar8x32_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    __m256 zv = _mm256_setzero_ps();
    __m256i permuteamask = _mm256_set_epi32(7,7,7,7,6,4,2,0);
    #pragma omp parallel for collapse(2) default(none) shared(B, oi_0, xi_0, IC, xi_1, OC, W_idx_OC, ICp1, W_idx_IC, oi_1, W_idx_X, W_idx_Y, W_val, W_scale, padding, M, OM, N, ON, X, dLdO, dLdX, dLdO_scale, dLdO_scale_v, x_scale_v, permuteamask, permutevar8x32_mask, zv, X_scale, dLdW_val)
    for (int b = 0; b < B; b++) {
        const int oi_2 = b * oi_0;
        const int xi_2 = b * xi_0;
        for (int ic = 0; ic < IC; ic++) {
            const int xi_3 = ic * xi_1;
            const int xi_4 = xi_2 + xi_3;
            for (int oc = 0; oc < OC; oc++) {
                const int oc_s = W_idx_OC[oc];
                const int offset = ICp1 * oc + ic;
                const int ic_s = oc_s + W_idx_IC[offset];
                const int ic_e = oc_s + W_idx_IC[offset + 1];
                const int oi_3 = oc * oi_1;
                const int oi_4 = oi_2 + oi_3;

                for (int si = ic_s; si < ic_e; si++) {
                    const uint8_t i = W_idx_X[si];
                    const uint8_t j = W_idx_Y[si];

                    //float x_scale_mul_dLdO_scale = X_scale * dLdO_scale;
                    const float v = (float) W_val[si] * W_scale;
                    const __m256 v0v = _mm256_set_ps(0., v, 0., v, 0., v, 0., v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);

                    for (int p = p_start; p < p_end; p++) {
                        const int oi_5 = p * ON;
                        const int oi_6 = oi_4 + oi_5;
                        const int xi_5 = (2 * p - pdmi) * N;
                        const int xi_6 = xi_4 + xi_5;
                        const int xi_7 = xi_6 - pdmj;

                        int8_t* X_ptr = X + xi_7;
                        int8_t*  dLdO_ptr = dLdO + oi_6;
                        float* dLdX_ptr = dLdX + xi_7;
                        int q = q_start;
                        for (; q < q_end - 7; q += 8) {
                            const __m128i ov_packed_i = _mm_loadu_si64(dLdO_ptr + q);
                            const __m128i x_0_packed_i = _mm_loadu_si64(X_ptr + 2 * q);
                            const __m128i x_1_packed_i = _mm_loadu_si64(X_ptr + 2 * q + 8);

                            // Unpack
                            __m256i ov_unpacked_i;
                            __m256i x_0_unpacked_i;
                            __m256i x_1_unpacked_i;
                            quantization::Quantization8Strategy::unpack8(ov_packed_i, ov_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(x_0_packed_i, x_0_unpacked_i);
                            quantization::Quantization8Strategy::unpack8(x_1_packed_i, x_1_unpacked_i);

                            // Convert from epi32 to ps
                            const __m256 ov_unpacked_f = _mm256_cvtepi32_ps(ov_unpacked_i);
                            const __m256 x_0_unpacked_f = _mm256_cvtepi32_ps(x_0_unpacked_i);
                            const __m256 x_1_unpacked_f = _mm256_cvtepi32_ps(x_1_unpacked_i);

                            // Dequantize
                            const __m256 ov_dq_f = _mm256_mul_ps(ov_unpacked_f, dLdO_scale_v);
                            const __m256 x_0_dq_f = _mm256_mul_ps(x_0_unpacked_f, x_scale_v);
                            const __m256 x_1_dq_f = _mm256_mul_ps(x_1_unpacked_f, x_scale_v);

                            // Permute for stride
                            const __m256 x_0_p = _mm256_permutevar8x32_ps(x_0_dq_f, permuteamask);
                            const __m256 x_1_p = _mm256_permutevar8x32_ps(x_1_dq_f, permuteamask);
                            const __m256 xv  = _mm256_insertf128_ps(x_0_p, _mm256_castps256_ps128(x_1_p), 1);

                            dwv = _mm256_fmadd_ps(ov_dq_f, xv, dwv);

                            const __m256 ov = _mm256_permutevar8x32_ps(ov_dq_f, permutevar8x32_mask);
                            const __m256 ov0 = _mm256_unpacklo_ps(ov, zv);
                            const __m256 ov1 = _mm256_unpackhi_ps(ov, zv);

                            const __m256 dxv0 = _mm256_loadu_ps(dLdX_ptr + 2 * q);
                            const __m256 dxv1 = _mm256_loadu_ps(dLdX_ptr + 2 * q + 8);

                            const __m256 dxv0_new = _mm256_fmadd_ps(ov0, v0v, dxv0);
                            const __m256 dxv1_new = _mm256_fmadd_ps(ov1, v0v, dxv1);

                            _mm256_storeu_ps(dLdX_ptr + 2 * q, dxv0_new);
                            _mm256_storeu_ps(dLdX_ptr + 2 * q + 8, dxv1_new);
                        }

                        for (; q < q_end; q++) {
                            float o = (float) dLdO_ptr[q] * dLdO_scale;
                            float x = (float) X_ptr[2 * q] * X_scale;

                            dLdX_ptr[2 * q] += v * o;
                            dw += o * x;
                        }
                    }
                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dwv = _mm256_hadd_ps(dwv, dwv);
                    dw += _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(dwv), _mm256_extractf128_ps(dwv, 1)));

                    #pragma omp atomic
                    dLdW_val[si] += dw;
                }
            }
        }
    }
#else
    sparse::quantized_sparse_conv2d_vectorized_backward_over_on_stride_2(
        B, IC, OC, M, N, K, W_nnz, padding, X, X_scale, W_idx_OC, W_idx_IC, W_idx_X,
        W_idx_Y, W_val, W_scale, dLdO, dLdO_scale, dLdX, dLdW_val
    );
#endif
}