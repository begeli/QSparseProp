#include "tensor_utils.h"

void transpose(float* X, float* XT, const int N, const int M, const int block_size) {
    #pragma omp parallel for default(none) shared(block_size, N, M, X, XT)
    for(int i = 0; i < N; i += block_size) {
        for(int j = 0; j < M; j += block_size) {
            int max_i2 = i + block_size < N ? i + block_size : N;
            int max_j2 = j + block_size < M ? j + block_size : M;
            for(int i2 = i; i2 < max_i2; i2 += 16) {
                for(int j2 = j; j2 < max_j2; j2 += 16) {
                    transpose_16x16(&X[i2 * M + j2], &XT[j2 * N + i2], M, N);
                }
            }
        }
    }
}

// Taken from https://gist.github.com/nihui/37d98b705a6a28911d77c502282b4748
void transpose_16x16(float* mat, float* matT, const int lda, const int ldb) {
    const __m512 r0_0 = _mm512_loadu_ps((void*) &mat[0]);
    const __m512 r1_0 = _mm512_loadu_ps((void*) &mat[1 * lda]);
    const __m512 r2_0 = _mm512_loadu_ps((void*) &mat[2 * lda]);
    const __m512 r3_0 = _mm512_loadu_ps((void*) &mat[3 * lda]);
    const __m512 r4_0 = _mm512_loadu_ps((void*) &mat[4 * lda]);
    const __m512 r5_0 = _mm512_loadu_ps((void*) &mat[5 * lda]);
    const __m512 r6_0 = _mm512_loadu_ps((void*) &mat[6 * lda]);
    const __m512 r7_0 = _mm512_loadu_ps((void*) &mat[7 * lda]);
    const __m512 r8_0 = _mm512_loadu_ps((void*) &mat[8 * lda]);
    const __m512 r9_0 = _mm512_loadu_ps((void*) &mat[9 * lda]);
    const __m512 ra_0 = _mm512_loadu_ps((void*) &mat[10 * lda]);
    const __m512 rb_0 = _mm512_loadu_ps((void*) &mat[11 * lda]);
    const __m512 rc_0 = _mm512_loadu_ps((void*) &mat[12 * lda]);
    const __m512 rd_0 = _mm512_loadu_ps((void*) &mat[13 * lda]);
    const __m512 re_0 = _mm512_loadu_ps((void*) &mat[14 * lda]);
    const __m512 rf_0 = _mm512_loadu_ps((void*) &mat[15 * lda]);

    const __m512 tmp0_0 = _mm512_unpacklo_ps(r0_0, r1_0);
    const __m512 tmp1_0 = _mm512_unpackhi_ps(r0_0, r1_0);
    const __m512 tmp2_0 = _mm512_unpacklo_ps(r2_0, r3_0);
    const __m512 tmp3_0 = _mm512_unpackhi_ps(r2_0, r3_0);
    const __m512 tmp4_0 = _mm512_unpacklo_ps(r4_0, r5_0);
    const __m512 tmp5_0 = _mm512_unpackhi_ps(r4_0, r5_0);
    const __m512 tmp6_0 = _mm512_unpacklo_ps(r6_0, r7_0);
    const __m512 tmp7_0 = _mm512_unpackhi_ps(r6_0, r7_0);
    const __m512 tmp8_0 = _mm512_unpacklo_ps(r8_0, r9_0);
    const __m512 tmp9_0 = _mm512_unpackhi_ps(r8_0, r9_0);
    const __m512 tmpa_0 = _mm512_unpacklo_ps(ra_0, rb_0);
    const __m512 tmpb_0 = _mm512_unpackhi_ps(ra_0, rb_0);
    const __m512 tmpc_0 = _mm512_unpacklo_ps(rc_0, rd_0);
    const __m512 tmpd_0 = _mm512_unpackhi_ps(rc_0, rd_0);
    const __m512 tmpe_0 = _mm512_unpacklo_ps(re_0, rf_0);
    const __m512 tmpf_0 = _mm512_unpackhi_ps(re_0, rf_0);

    const __m512 tmpg = _mm512_shuffle_ps(tmp0_0, tmp2_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmph = _mm512_shuffle_ps(tmp0_0, tmp2_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpi = _mm512_shuffle_ps(tmp1_0, tmp3_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpj = _mm512_shuffle_ps(tmp1_0, tmp3_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpk = _mm512_shuffle_ps(tmp4_0, tmp6_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpl = _mm512_shuffle_ps(tmp4_0, tmp6_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpm = _mm512_shuffle_ps(tmp5_0, tmp7_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpn = _mm512_shuffle_ps(tmp5_0, tmp7_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpo = _mm512_shuffle_ps(tmp8_0, tmpa_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpp = _mm512_shuffle_ps(tmp8_0, tmpa_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpq = _mm512_shuffle_ps(tmp9_0, tmpb_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpr = _mm512_shuffle_ps(tmp9_0, tmpb_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmps = _mm512_shuffle_ps(tmpc_0, tmpe_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpt = _mm512_shuffle_ps(tmpc_0, tmpe_0, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 tmpu = _mm512_shuffle_ps(tmpd_0, tmpf_0, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 tmpv = _mm512_shuffle_ps(tmpd_0, tmpf_0, _MM_SHUFFLE(3, 2, 3, 2));

    const __m512 tmp0_1 = _mm512_shuffle_f32x4(tmpg, tmpk, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp1_1 = _mm512_shuffle_f32x4(tmpo, tmps, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp2_1 = _mm512_shuffle_f32x4(tmph, tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp3_1 = _mm512_shuffle_f32x4(tmpp, tmpt, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp4_1 = _mm512_shuffle_f32x4(tmpi, tmpm, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp5_1 = _mm512_shuffle_f32x4(tmpq, tmpu, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp6_1 = _mm512_shuffle_f32x4(tmpj, tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp7_1 = _mm512_shuffle_f32x4(tmpr, tmpv, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 tmp8_1 = _mm512_shuffle_f32x4(tmpg, tmpk, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmp9_1 = _mm512_shuffle_f32x4(tmpo, tmps, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpa_1 = _mm512_shuffle_f32x4(tmph, tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpb_1 = _mm512_shuffle_f32x4(tmpp, tmpt, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpc_1 = _mm512_shuffle_f32x4(tmpi, tmpm, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpd_1 = _mm512_shuffle_f32x4(tmpq, tmpu, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpe_1 = _mm512_shuffle_f32x4(tmpj, tmpn, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 tmpf_1 = _mm512_shuffle_f32x4(tmpr, tmpv, _MM_SHUFFLE(3, 1, 3, 1));

    const __m512 r0_1 = _mm512_shuffle_f32x4(tmp0_1, tmp1_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r1_1 = _mm512_shuffle_f32x4(tmp2_1, tmp3_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r2_1 = _mm512_shuffle_f32x4(tmp4_1, tmp5_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r3_1 = _mm512_shuffle_f32x4(tmp6_1, tmp7_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r4_1 = _mm512_shuffle_f32x4(tmp8_1, tmp9_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r5_1 = _mm512_shuffle_f32x4(tmpa_1, tmpb_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r6_1 = _mm512_shuffle_f32x4(tmpc_1, tmpd_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r7_1 = _mm512_shuffle_f32x4(tmpe_1, tmpf_1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r8_1 = _mm512_shuffle_f32x4(tmp0_1, tmp1_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 r9_1 = _mm512_shuffle_f32x4(tmp2_1, tmp3_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 ra_1 = _mm512_shuffle_f32x4(tmp4_1, tmp5_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 rb_1 = _mm512_shuffle_f32x4(tmp6_1, tmp7_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 rc_1 = _mm512_shuffle_f32x4(tmp8_1, tmp9_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 rd_1 = _mm512_shuffle_f32x4(tmpa_1, tmpb_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 re_1 = _mm512_shuffle_f32x4(tmpc_1, tmpd_1, _MM_SHUFFLE(3, 1, 3, 1));
    const __m512 rf_1 = _mm512_shuffle_f32x4(tmpe_1, tmpf_1, _MM_SHUFFLE(3, 1, 3, 1));

    _mm512_storeu_ps(&matT[0], r0_1);
    _mm512_storeu_ps(&matT[1 * ldb], r1_1);
    _mm512_storeu_ps(&matT[2 * ldb], r2_1);
    _mm512_storeu_ps(&matT[3 * ldb], r3_1);
    _mm512_storeu_ps(&matT[4 * ldb], r4_1);
    _mm512_storeu_ps(&matT[5 * ldb], r5_1);
    _mm512_storeu_ps(&matT[6 * ldb], r6_1);
    _mm512_storeu_ps(&matT[7 * ldb], r7_1);
    _mm512_storeu_ps(&matT[8 * ldb], r8_1);
    _mm512_storeu_ps(&matT[9 * ldb], r9_1);
    _mm512_storeu_ps(&matT[10 * ldb], ra_1);
    _mm512_storeu_ps(&matT[11 * ldb], rb_1);
    _mm512_storeu_ps(&matT[12 * ldb], rc_1);
    _mm512_storeu_ps(&matT[13 * ldb], rd_1);
    _mm512_storeu_ps(&matT[14 * ldb], re_1);
    _mm512_storeu_ps(&matT[15 * ldb], rf_1);
}
