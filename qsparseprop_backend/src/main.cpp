#include <iostream>
#include <immintrin.h>
#include <omp.h>

#include <cstring>
#include <cmath>

#include "src/quantization/standard_quantization8_strategy.h"
#include "src/quantization/quantization8_strategy.h"

void test();
void test_quantization();
void scalar_quantization_test();
void preprocessor_test();
void test_quantization_strategy();
void test_truncate();
void test_inf();
//void test_log2();

int main() {
  //  std::cout << "Hello, World!" << std::endl;

    //char sentence[] = "This is a sentence.";
    //char mem[19];

    //strcpy(mem, sentence);
    //printf("'%s' is the resulting string.\n", mem);
    float inf = 1.0f / -0.0f;
    std::cout << inf << std::endl;
    float maybe_inf = -1.0f / 0.0f;
    std::cout << (maybe_inf == inf) << std::endl;
    test_inf();
    test_quantization();
    std::cout << "Testing quantization " << std::endl;
    test_quantization_strategy();
    test_truncate();
    omp_set_num_threads(4);
    int nthr = -1;
#pragma omp parallel shared(nthr)
    {
        nthr = omp_get_num_threads();
    }
    std::cout << nthr << std::endl;
//#pragma omp parallel
//    {
//        std::cout << "Hello world! but with omp" << std::endl;
//    }

    int8_t arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m128i pack8 = _mm_loadu_si64(arr);
    __m256i out;
    quantization::Quantization8Strategy::unpack8(pack8, out);
    int arr2[8];
    _mm256_storeu_epi32(arr2, out);

    for (int i = 0; i < 8; i++) {
        std::cout << arr2[i] << std::endl;
    }
    return 0;
}

void test_set_const();

void test_inf() {
    __m512 inf = _mm512_set1_ps(1.0f / 0.0f);
    float arr[16];
    _mm512_storeu_ps(arr, inf);
    std::cout << arr[0] << std::endl;
}

/*void test_log2() {
    __m512 all2s = _mm512_set1_ps(8.0f);
    __m512 log2s = _mm512_log2_ps(all2s);
    float arr[16];
    _mm512_storeu_ps(arr, log2s);
    std::cout << arr[0] << std::endl;
}*/

void test_truncate() {
    int arr[16];
    __m512 float_vec = _mm512_set1_ps(0.7f);
    __m512i converted_vec = _mm512_cvtps_epi32(float_vec);
    _mm512_storeu_epi32(arr, converted_vec);

    std::cout << "Conversion is " << arr[0] << std::endl;
}

void preprocessor_test() {
#ifdef TEST
    std::cout << "Defined outside works" << std::endl;
#else
    std::cout << "Defined outside doesn't work" << std::endl;
#endif
}

void test() {
    __m256 vector_type = _mm256_set_ps(0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
    std::cout << "Loaded float vector" << std::endl;

    __m512 big_vector_type = _mm512_set1_ps(0.0f);
    std::cout << "Loaded biig vector" << std::endl;

    __m256 sum = _mm256_add_ps(vector_type, vector_type);
    float sumArr[8];
    _mm256_store_ps(sumArr, sum);

    for (int i = 0; i < 8; i++) {
        std::cout << sumArr[i] << std::endl;
    }

    __m512 vector_type_2 = _mm512_set1_ps(0.2f);
    __m512 vector_type_3 = _mm512_set1_ps(0.2f);
    __m512 sum512 = _mm512_add_ps(vector_type_2, vector_type_3);
    float sumArr2[16];
    _mm512_storeu_ps(sumArr2, sum512);
    for (int i = 0; i < 16; i++) {
        std::cout << sumArr2[i] << std::endl;
    }
    __m512 vec = _mm512_set1_ps(0.5);
    _mm512_cvttps_epi32(vec);

    test_quantization_strategy();
}

void test_quantization_strategy() {
    /*float arr1[32] = {
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f, 5.0f,
            5.0f, 5.0f, 5.0f
    };*/
    float arr1[32] = {
            -16.5f, -13.5f, -12.5f, -10.5f,
            -8.5f, -6.5f, -4.5f, -2.5f,
            -0.5f, 1.5f, 3.5f, 5.5f,
            7.5f, 9.5f, 11.5f, 13.5f,
            -16.5f, -13.5f, -12.5f, -10.5f,
            -8.5f, -6.5f, -4.5f, -2.5f,
            -0.5f, 1.5f, 3.5f, 5.5f,
            7.5f, 9.5f, 11.5f, 13.5f,
    };
    int8_t res[32];
    union quantization::Quantization_Input<int8_t> input
            = {.std_quantization_input={arr1, res, 32, 0.0f, 0.0f}};
    quantization::StandardQuantization8Strategy q_strategy
        = quantization::StandardQuantization8Strategy(-128, 127);
    q_strategy.quantize(input);
    std::cout << "Const: " << input.std_quantization_input.dequantization_const << std::endl;
    for (int i = 0; i < 32; i++) {
        std::cout << (int) res[i] << std::endl;
    }
    std::cout << input.std_quantization_input.scale << std::endl;

    float arr2[32];
    union quantization::Quantization_Input<int8_t> restore_input
            = {.std_quantization_input={arr2, res, 32, input.std_quantization_input.scale, input.std_quantization_input.dequantization_const}};
    q_strategy.restore(restore_input);
    for (int i = 0; i < 32; i++) {
        std::cout << arr2[i] << std::endl;
    }
}

void test_quantization() {
    float arr1[16] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f
    };
    __m512 val_vec_0 = _mm512_loadu_ps(arr1);/*_mm512_set_ps(
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f
    );*/
    __m512 val_vec_1 = _mm512_loadu_ps(arr1);/*_mm512_set_ps(
            -16.5f, -13.5f, -12.5f, -10.5f,
            -8.5f, -6.5f, -4.5f, -2.5f,
            -0.5f, 1.5f, 3.5f, 5.5f,
            7.5f, 9.5f, 11.5f, 13.5f
    );*/
    /*
    __m512 val_vec_1 =_mm512_set_ps(
        -103.4f, 2.45f, -45.04f, 54.2f,
        2.4f, 0.0f, 152.5f, -23.5,
        8.1f, 24.5f, -12.56f, 72.5f,
        -32.0f, 67.2f, -39.45f, 0.0f
    );*/

    __m512 min_vec_1 = _mm512_set1_ps(-16.5f);
    __m512 lower_bound_vec = _mm512_set1_ps(-128);
    __m512 upper_bound_vec = _mm512_set1_ps(127);
    __m512 value_range = _mm512_set1_ps(13.5f - (-16.5f));
    __m512 bit_range = _mm512_sub_ps(upper_bound_vec, lower_bound_vec);//_mm512_set1_ps(upperBound - lowerBound);
    __m512 scale = _mm512_div_ps(value_range, bit_range);
    __m512 rcp_scale = _mm512_div_ps(bit_range, _mm512_add_ps(value_range, _mm512_set1_ps(0.001f))); // TODO: changed this
    __m512 quantization_const_0 = _mm512_fmsub_ps(lower_bound_vec, scale, min_vec_1);
    __m512 quantization_const_1 = _mm512_mul_ps(rcp_scale, quantization_const_0);

    __m512 rnd_0 = _mm512_setzero_ps();
    __m512 rnd_1 = _mm512_setzero_ps();

    // Quantize
    __m512 normalized_vec_0 = _mm512_fmadd_ps(rcp_scale, val_vec_0, quantization_const_1);
    __m512 normalized_vec_1 = _mm512_fmadd_ps(rcp_scale, val_vec_1, quantization_const_1);
    __m512 normalized_rnd_vec_0 = _mm512_add_ps(normalized_vec_0, rnd_0);
    __m512 normalized_rnd_vec_1 = _mm512_add_ps(normalized_vec_1, rnd_1);

    __m512 lower_clamped_vec_0 = _mm512_max_ps(normalized_rnd_vec_0, lower_bound_vec);
    __m512 upper_clamped_vec_0 = _mm512_min_ps(lower_clamped_vec_0, upper_bound_vec);
    __m512 lower_clamped_vec_1 = _mm512_max_ps(normalized_rnd_vec_1, lower_bound_vec);
    __m512 upper_clamped_vec_1 = _mm512_min_ps(lower_clamped_vec_1, upper_bound_vec);

    __m512i quantized_vec_0 = _mm512_cvttps_epi32(upper_clamped_vec_0);
    __m512i quantized_vec_1 = _mm512_cvttps_epi32(upper_clamped_vec_1);

    // Pack
    // Step 1: break the 512bit vectors into two
    int v1[16];
    int v2[8];
    int v3[8];
    __m256i q_vec_0_lo = _mm512_castsi512_si256(quantized_vec_0);
    __m256i q_vec_0_hi = _mm512_extracti32x8_epi32(quantized_vec_0, 0x1);
    __m256i q_vec_1_lo = _mm512_castsi512_si256(quantized_vec_1);
    __m256i q_vec_1_hi = _mm512_extracti32x8_epi32(quantized_vec_1, 0x1);

    _mm512_storeu_epi32(v1, quantized_vec_0);
    _mm256_store_epi32(v2, q_vec_0_lo);
    _mm256_store_epi32(v3, q_vec_0_hi);
    //std::cout << "full-----------------------------------" << std::endl;
    for (int i = 0; i < 16; i++) {
    //    std::cout << v1[i] << std::endl;
    }
    //std::cout << "first-----------------------------------" << std::endl;
    for (int i = 0; i < 8; i++) {
    //    std::cout << v2[i] << std::endl;
    }
    //std::cout << "second-----------------------------------" << std::endl;
    for (int i = 0; i < 8; i++) {
    //    std::cout << v3[i] << std::endl;
    }

    // Step 2: Shift the vectors
    __m256i left0 = _mm256_slli_epi32(q_vec_0_lo, 24);
    __m256i left1 = _mm256_slli_epi32(q_vec_0_hi, 24);
    __m256i left2 = _mm256_slli_epi32(q_vec_1_lo, 24);
    __m256i left3 = _mm256_slli_epi32(q_vec_1_hi, 24);

    __m256i right0 = _mm256_srli_epi32(left0, 24);
    __m256i right1 = _mm256_srli_epi32(left1, 16);
    __m256i right2 = _mm256_srli_epi32(left2, 24);
    __m256i right3 = _mm256_srli_epi32(left3, 16);

    __m256i pack16_0 = _mm256_or_si256(right0, right1);
    __m256i pack16_1 = _mm256_or_si256(right2, right3);

    __m256i interleave_lo_0 = _mm256_permute2f128_si256(pack16_0, pack16_1, 0x20);
    __m256i interleave_hi_0 = _mm256_permute2f128_si256(pack16_0, pack16_1, 0x31);

    __m256i _mm256_8bit_perm_lo = _mm256_setr_epi8 (
            0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15,
            0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15
    );
    __m256i _mm256_8bit_perm_hi = _mm256_setr_epi8 (
            2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13,
            2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13
    );
    __m256i permute_lo_0 = _mm256_shuffle_epi8(interleave_lo_0, _mm256_8bit_perm_lo);
    __m256i permute_hi_0 = _mm256_shuffle_epi8(interleave_hi_0, _mm256_8bit_perm_hi);
    __m256i pack8_lo = _mm256_or_si256(permute_lo_0, permute_hi_0);

    int8_t quantized_vals[32];

    //_mm512_storeu_epi32(quantized_vals, quantized_vec_0);
    _mm256_storeu_si256((__m256i *)(quantized_vals +  0), pack8_lo);
    for (int i = 0; i < 32; i++) {
        std::cout << (int)quantized_vals[i] << std::endl;
    }
}

void scalar_quantization_test() {
    float values[16] = {
        -16.5f, -13.5f, -12.5f, -10.5f,
        -8.5f, -6.5f, -4.5f, -2.5f,
        -0.5f, 1.5f, 3.5f, 5.5f,
        7.5f, 9.5f, 11.5f, 13.5f
    };

    float min = -16.5f;
    float max = 13.5f;
    float lower_bound = -128.0f;
    float upper_bound = 127.0f;
    float value_range = max - min;
    float bit_range = upper_bound - lower_bound;
    float scale = value_range / bit_range;
    float rcp_scale = 1.0f / scale;
    float quantization_const = rcp_scale * (lower_bound * scale - min);

    for (int i = 0; i < 16; i++) {
        float normalized = rcp_scale * values[i] + quantization_const;
        float clamped = std::min(std::max(normalized, lower_bound), upper_bound);
        int8_t rounded_down = (int8_t) floorf(clamped);
        std::cout << "normalized: " << normalized << " clamped: " << clamped << " rounded_down: " << (int) rounded_down << std::endl;
    }

    float scaled_rcp_max = 127.0f / 16.5f;
    for (int i = 0; i < 16; i++) {
        float f_value = values[i];
        int8_t sign = (int8_t) 1 + ((int8_t) (*(int32_t *) &f_value >> 31) << 1);
        uint32_t abs_value = 0x7FFFFFFFU & *(uint32_t *) &f_value;
        float f_value_abs = *(float *) &abs_value;
        int8_t   q_value_abs = (int8_t) floorf(scaled_rcp_max * f_value_abs);
        std::cout << "f_value: " << f_value << " sign: " << (int) sign << " abs_value: " << abs_value <<
            " f_value_abs: " << f_value_abs << " q_value_abs: " << (int) q_value_abs << std::endl;
        std::cout << (int) (q_value_abs * sign) << std::endl;

    }
}