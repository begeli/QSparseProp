#ifndef QSPARSEPROP_QUANTIZATION8_STRATEGY_H
#define QSPARSEPROP_QUANTIZATION8_STRATEGY_H

#include "quantization_strategy.h"

namespace quantization {
    class Quantization8Strategy: public quantization::QuantizationStrategy<int8_t>{
    public:
        Quantization8Strategy() = default;

        virtual void quantize(union Quantization_Input<int8_t>& input) = 0;
        virtual void quantize_parallel(union Quantization_Input<int8_t>& input) = 0;
        virtual void restore(union Quantization_Input<int8_t>& input) = 0;
        virtual void restore_parallel(union Quantization_Input<int8_t>& input) = 0;

        static inline void pack32(
            __m256i q_vec_0_lo, __m256i q_vec_0_hi, __m256i q_vec_1_lo, __m256i q_vec_1_hi, __m256i& pack8
        ) {
            const __m256i left0 = _mm256_slli_epi32(q_vec_0_lo, 24);
            const __m256i left1 = _mm256_slli_epi32(q_vec_0_hi, 24);
            const __m256i left2 = _mm256_slli_epi32(q_vec_1_lo, 24);
            const __m256i left3 = _mm256_slli_epi32(q_vec_1_hi, 24);

            const __m256i right0 = _mm256_srli_epi32(left0, 24);
            const __m256i right1 = _mm256_srli_epi32(left1, 16);
            const __m256i right2 = _mm256_srli_epi32(left2, 24);
            const __m256i right3 = _mm256_srli_epi32(left3, 16);

            // Combine the shifted halves
            const __m256i pack16_0 = _mm256_or_si256(right0, right1);
            const __m256i pack16_1 = _mm256_or_si256(right2, right3);

            const __m256i interleave_lo_0 = _mm256_permute2f128_si256(pack16_0, pack16_1, 0x20);
            const __m256i interleave_hi_0 = _mm256_permute2f128_si256(pack16_0, pack16_1, 0x31);

            const __m256i permute_lo_0 = _mm256_shuffle_epi8(interleave_lo_0, _mm256_8bit_perm_lo);
            const __m256i permute_hi_0 = _mm256_shuffle_epi8(interleave_hi_0, _mm256_8bit_perm_hi);
            pack8 = _mm256_or_si256(permute_lo_0, permute_hi_0);
        }

        static inline void unpack64(
            __m256i input_0, __m256i input_1,
            __m256i& out_0, __m256i& out_1, __m256i& out_2, __m256i& out_3,
            __m256i& out_4, __m256i& out_5, __m256i& out_6, __m256i& out_7
        ) {
            const __m256i q_val_vec_switched_0 = _mm256_permute2f128_si256(input_0, input_0, 0x21);
            const __m256i q_val_vec_switched_1 = _mm256_permute2f128_si256(input_1, input_1, 0x21);

            const __m256i q_val_half_0_0 = _mm256_shuffle_epi8(input_0, __mm512_8bit_restore_perm_lo);
            const __m256i q_val_half_0_1 = _mm256_shuffle_epi8(q_val_vec_switched_0, __mm512_8bit_restore_perm_hi);
            const __m256i q_val_half_1_0 = _mm256_shuffle_epi8(input_1, __mm512_8bit_restore_perm_lo);
            const __m256i q_val_half_1_1 = _mm256_shuffle_epi8(q_val_vec_switched_1, __mm512_8bit_restore_perm_hi);

            const __m256i q_val_full_0 = _mm256_or_si256(q_val_half_0_0, q_val_half_0_1);
            const __m256i q_val_full_1 = _mm256_or_si256(q_val_half_1_0, q_val_half_1_1);

            const __m256i left_0 = _mm256_slli_epi32(q_val_full_0, 24);
            const __m256i left_1 = _mm256_slli_epi32(q_val_full_0, 16);
            const __m256i left_2 = _mm256_slli_epi32(q_val_full_0, 8);
            const __m256i left_3 = _mm256_slli_epi32(q_val_full_0, 0);
            const __m256i left_4 = _mm256_slli_epi32(q_val_full_1, 24);
            const __m256i left_5 = _mm256_slli_epi32(q_val_full_1, 16);
            const __m256i left_6 = _mm256_slli_epi32(q_val_full_1, 8);
            const __m256i left_7 = _mm256_slli_epi32(q_val_full_1, 0);

            out_0 = _mm256_srai_epi32(left_0, 24);
            out_1 = _mm256_srai_epi32(left_1, 24);
            out_2 = _mm256_srai_epi32(left_2, 24);
            out_3 = _mm256_srai_epi32(left_3, 24);
            out_4 = _mm256_srai_epi32(left_4, 24);
            out_5 = _mm256_srai_epi32(left_5, 24);
            out_6 = _mm256_srai_epi32(left_6, 24);
            out_7 = _mm256_srai_epi32(left_7, 24);
        }

        // Assumption, this code assumes only the first 64 bits of the input vector contains 8-bit integers
        // Unpack 8 integers into a single 256-bit vector
        static inline void unpack8(__m128i input, __m256i& out) {
            const __m128i lower = _mm_shuffle_epi8(input, __mm_8bit_restore_shuffle_lo);
            const __m128i lower_shifted = _mm_srai_epi32(lower, 24);
            const __m128i upper = _mm_shuffle_epi8(input, __mm_8bit_restore_shuffle_hi);
            const __m128i upper_shifted = _mm_srai_epi32(upper, 24);
            out = _mm256_inserti32x4(_mm256_castsi128_si256(lower_shifted), upper_shifted, 1);
        }
    };
}

#endif //QSPARSEPROP_QUANTIZATION8_STRATEGY_H
