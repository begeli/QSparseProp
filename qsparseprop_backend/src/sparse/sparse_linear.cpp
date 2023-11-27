#include "sparse_linear.h"

void sparse::sparse_linear_vectorized_forward(
    int B, int M, int N, int W_nnz,
    float* X, int* W_idx_N, int* W_idx_M,
    float* W_val, float* O
) {
#pragma omp parallel
    {
#pragma omp for
        for(int i = 0; i < N; i++){
            int k = W_idx_N[i];
            for(; k < W_idx_N[i+1]; k++){
                int idx = W_idx_M[k];
                __m256 v = _mm256_set1_ps(W_val[k]);
                int j = 0;

                for(; j < B-7; j+=8){
                    __m256 x = _mm256_loadu_ps(X + (idx * B + j));
                    __m256 o = _mm256_loadu_ps(O + (i * B + j));

                    __m256 r = _mm256_fmadd_ps(x,v,o);

                    _mm256_storeu_ps(O + (i * B + j), r);
                }

                for(; j < B; j++){
                    O[i * B + j] += W_val[k] * X[idx * B + j];
                }
            }
        }
    }
}

void sparse::sparse_linear_vectorized_backward(
    int B, int M, int N, int W_nnz, float* X,
    int* W_idx_N, int* W_idx_M,float* W_val,
    float* dLdO, float* dLdX, float* dLdW_val
) {
#pragma omp parallel
    {
        for(int i = 0; i < N; i++){
#pragma omp for
            for(int j = W_idx_N[i]; j < W_idx_N[i+1]; j++){
                int r = W_idx_M[j];
                float sv = W_val[j];
                __m256 v = _mm256_set1_ps(W_val[j  ]);
                float sacc = 0;
                __m256 acc = _mm256_setzero_ps();

                int k = 0;
                for(; k < B-7; k+=8){
                    __m256 dx0 = _mm256_loadu_ps(dLdX + (r * B + k));
                    __m256 x0 = _mm256_loadu_ps(X + (r * B + k));
                    __m256 do0 = _mm256_loadu_ps(dLdO + (i * B + k));
                    __m256 s0  = _mm256_fmadd_ps(v, do0, dx0);
                    acc = _mm256_fmadd_ps(do0,x0,acc);
                    _mm256_storeu_ps(dLdX + (r * B + k), s0);
                }

                //cleanup
                for(; k < B; k++){
                    dLdX[r*B+k] += sv * dLdO[i * B + k];
                    sacc += dLdO[i*B + k] * X[r*B + k];
                }

                //reduce sum
                const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
                const __m128 loQuad0 = _mm256_castps256_ps128(acc);
                const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

                dLdW_val[j] = sacc + _mm_cvtss_f32(sum0);
            }
        }
    }
}