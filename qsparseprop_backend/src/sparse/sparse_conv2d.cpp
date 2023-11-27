#include "sparse_conv2d.h"

// ====================================== Stride 1 ===========================================
void sparse::sparse_conv2d_vectorized_forward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val, float* O
) {

    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;

#pragma omp parallel
    {

#pragma omp for
        for (int oc = 0; oc < OC; oc++){
            for (int ic = 0; ic < IC; ic++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(W_val[si]);

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);

                    for (int po = p_start, px = p_start - padding + i; po < p_end; po++, px++) {
                        int qo = q_start, qx = q_start - padding + j;
                        for (; qo < q_end-3; qo+=4, qx+=4) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o0 = _mm256_loadu_ps(O + Oi);
                                const __m256 o1 = _mm256_loadu_ps(O + Oi + B);
                                const __m256 o2 = _mm256_loadu_ps(O + Oi + 2 * B);
                                const __m256 o3 = _mm256_loadu_ps(O + Oi + 3 * B);
                                const __m256 x0 = _mm256_loadu_ps(X + Xi);
                                const __m256 x1 = _mm256_loadu_ps(X + Xi + B);
                                const __m256 x2 = _mm256_loadu_ps(X + Xi + 2 * B);
                                const __m256 x3 = _mm256_loadu_ps(X + Xi + 3 * B);

                                const __m256 r0 = _mm256_fmadd_ps(x0,vv,o0);
                                const __m256 r1 = _mm256_fmadd_ps(x1,vv,o1);
                                const __m256 r2 = _mm256_fmadd_ps(x2,vv,o2);
                                const __m256 r3 = _mm256_fmadd_ps(x3,vv,o3);

                                _mm256_storeu_ps(O + Oi, r0);
                                _mm256_storeu_ps(O + Oi + B, r1);
                                _mm256_storeu_ps(O + Oi + 2 * B, r2);
                                _mm256_storeu_ps(O + Oi + 3 * B, r3);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                        for (; qo < q_end; qo++, qx++) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o = _mm256_loadu_ps(O + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);

                                const __m256 r = _mm256_fmadd_ps(x,vv,o);

                                _mm256_storeu_ps(O + Oi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                    }
                }
            }
        }
    }
}

void sparse::sparse_conv2d_vectorized_backward_stride_1(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val, float* dLdO, float* dLdX, float* dLdW_val
) {
    const int OM = M + 2 * padding - K + 1;
    const int ON = N + 2 * padding - K + 1;

#pragma omp parallel
    {

#pragma omp for reduction(+:dLdW_val[:W_nnz])
        for (int ic = 0; ic < IC; ic++){
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max(pdmi, 0);
                    const int p_end = std::min(pdmi + M, OM);
                    const int q_start = std::max(pdmj, 0);
                    const int q_end = std::min(pdmj + N, ON);

                    for (int po = p_start, px = p_start - padding + i; po < p_end; po++, px++) {
                        int qo = q_start, qx = q_start - padding + j;
                        for (; qo < q_end; qo++, qx++) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o = _mm256_loadu_ps(dLdO + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);
                                const __m256 dx = _mm256_loadu_ps(dLdX + Xi);

                                const __m256 r = _mm256_fmadd_ps(o,vv,dx);
                                dwv = _mm256_fmadd_ps(o, x, dwv);

                                _mm256_storeu_ps(dLdX + Xi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                float o = dLdO[Oi];
                                float x = X[Xi];

                                dLdX[Xi] += o * v;

                                dw += o * x;
                            }
                        }
                    }

                    const __m128 hiQuad0 = _mm256_extractf128_ps(dwv, 1);
                    const __m128 loQuad0 = _mm256_castps256_ps128(dwv);
                    const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                    const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                    const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                    const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                    const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

                    dLdW_val[si] += dw + _mm_cvtss_f32(sum0);
                }
            }
        }
    }
}

// ====================================== Stride 2 ===========================================

void sparse::sparse_conv2d_vectorized_forward_stride_2(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val, float* O
) {
    const int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    const int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);

#pragma omp parallel
    {

#pragma omp for
        for (int oc = 0; oc < OC; oc++){
            for (int ic = 0; ic < IC; ic++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(W_val[si]);

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);

                    for (int po = p_start, px = 2 * p_start - padding + i; po < p_end; po++, px+=2) {
                        int qo = q_start, qx = 2 * q_start - padding + j;
                        for (; qo < q_end-3; qo+=4, qx+=8) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o0 = _mm256_loadu_ps(O + Oi);
                                const __m256 o1 = _mm256_loadu_ps(O + Oi + B);
                                const __m256 o2 = _mm256_loadu_ps(O + Oi + 2 * B);
                                const __m256 o3 = _mm256_loadu_ps(O + Oi + 3 * B);
                                const __m256 x0 = _mm256_loadu_ps(X + Xi);
                                const __m256 x1 = _mm256_loadu_ps(X + Xi + 2 * B);
                                const __m256 x2 = _mm256_loadu_ps(X + Xi + 4 * B);
                                const __m256 x3 = _mm256_loadu_ps(X + Xi + 6 * B);

                                const __m256 r0 = _mm256_fmadd_ps(x0,vv,o0);
                                const __m256 r1 = _mm256_fmadd_ps(x1,vv,o1);
                                const __m256 r2 = _mm256_fmadd_ps(x2,vv,o2);
                                const __m256 r3 = _mm256_fmadd_ps(x3,vv,o3);

                                _mm256_storeu_ps(O + Oi, r0);
                                _mm256_storeu_ps(O + Oi + B, r1);
                                _mm256_storeu_ps(O + Oi + 2 * B, r2);
                                _mm256_storeu_ps(O + Oi + 3 * B, r3);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }

                        for (; qo < q_end; qo++, qx+=2) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;


                                const __m256 o = _mm256_loadu_ps(O + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);

                                const __m256 r = _mm256_fmadd_ps(x,vv,o);

                                _mm256_storeu_ps(O + Oi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                O[Oi] += X[Xi] * v;
                            }
                        }
                    }
                }
            }
        }
    }
}

void sparse::sparse_conv2d_vectorized_backward_stride_2(
    int B, int IC, int OC, int M, int N, int K, int W_nnz, int padding,
    float* X, int* W_idx_OC, int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val, float* dLdO, float* dLdX, float* dLdW_val
) {

    const int OM = (int) ceil((float) (M + 2 * padding - K + 1) / 2);
    const int ON = (int) ceil((float) (N + 2 * padding - K + 1) / 2);

#pragma omp parallel
    {

#pragma omp for reduction(+:dLdW_val[:W_nnz])
        for (int ic = 0; ic < IC; ic++){
            for (int oc = 0; oc < OC; oc++){
                int oc_s = W_idx_OC[oc];
                int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
                int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

                for (int si = ic_s; si < ic_e; si++) {
                    uint8_t i = W_idx_X[si];
                    uint8_t j = W_idx_Y[si];

                    float v = W_val[si];
                    __m256 vv = _mm256_set1_ps(v);
                    __m256 dwv = _mm256_setzero_ps();
                    float dw = 0;

                    const int pdmi = padding - i;
                    const int pdmj = padding - j;
                    const int p_start = std::max((int) ceil((float) pdmi / 2.0), 0);
                    const int p_end = std::min((int) floor((float) (pdmi + M - 1) / 2) + 1, OM);
                    const int q_start = std::max((int) ceil((float) pdmj / 2.0), 0);
                    const int q_end = std::min((int) floor((float) (pdmj + N - 1) / 2.0) + 1, ON);


                    for (int po = p_start, px = 2 * p_start - padding + i; po < p_end; po++, px+=2) {
                        int qo = q_start, qx = 2 * q_start - padding + j;
                        for (; qo < q_end; qo++, qx+=2) {
                            int b = 0;
                            for (; b < B-7; b+=8) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                const __m256 o = _mm256_loadu_ps(dLdO + Oi);
                                const __m256 x = _mm256_loadu_ps(X + Xi);
                                const __m256 dx = _mm256_loadu_ps(dLdX + Xi);

                                const __m256 r = _mm256_fmadd_ps(o,vv,dx);
                                dwv = _mm256_fmadd_ps(o, x, dwv);

                                _mm256_storeu_ps(dLdX + Xi, r);
                            }
                            for (; b < B; b++) {
                                int Xi = ic * B * M * N + px * N * B + qx * B + b;
                                int Oi = oc * OM * ON * B + po * ON * B + qo * B + b;

                                float o = dLdO[Oi];
                                float x = X[Xi];

                                dLdX[Xi] += o * v;

                                dw += o * x;
                            }
                        }
                    }

                    const __m128 hiQuad0 = _mm256_extractf128_ps(dwv, 1);
                    const __m128 loQuad0 = _mm256_castps256_ps128(dwv);
                    const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                    const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                    const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                    const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                    const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

                    dLdW_val[si] += dw + _mm_cvtss_f32(sum0);
                }
            }
        }
    }
}