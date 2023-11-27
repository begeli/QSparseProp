#include "simdxorshift128plus.h"

// TODO: Should I make these functions inline functions and carry them to the header file.
/* used by xorshift128plus_jump_onkeys */
static void xorshift128plus_onkeys(uint64_t * ps0, uint64_t * ps1) {
    uint64_t s1 = *ps0;
    const uint64_t s0 = *ps1;
    *ps0 = s0;
    s1 ^= s1 << 23; // a
    *ps1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
}

/* used by avx_xorshift128plus_init */
static void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2, uint64_t * output1, uint64_t * output2) {
    static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & 1ULL << b) {
                s0 ^= in1;
                s1 ^= in2;
            }
            xorshift128plus_onkeys(&in1, &in2);
        }
    output1[0] = s0;
    output2[0] = s1;
}

void avx_xorshift128plus_init(uint64_t key1, uint64_t key2, __m256i &part1, __m256i &part2) {
    uint64_t S0[4];
    uint64_t S1[4];
    S0[0] = key1;
    S1[0] = key2;
    xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
    xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
    xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
    part1 = _mm256_loadu_si256((const __m256i *) S0);
    part2 = _mm256_loadu_si256((const __m256i *) S1);
}

/*
 Return a 256-bit random "number"
 */
__m256i avx_xorshift128plus(__m256i &part1, __m256i &part2) {
    __m256i s1 = part1;
    const __m256i s0 = part2;
    part1 = part2;
    s1 = _mm256_xor_si256(part2, _mm256_slli_epi64(part2, 23));
    part2 = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(s1, s0),_mm256_srli_epi64(s1, 18)),
        _mm256_srli_epi64(s0, 5)
    );
    return _mm256_add_epi64(part2, s0);
}

void avx512_xorshift128plus_init(uint64_t key1, uint64_t key2, __m512i &part1, __m512i &part2) {
    uint64_t S0[8];
    uint64_t S1[8];
    S0[0] = key1;
    S1[0] = key2;

    xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
    xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
    xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
    xorshift128plus_jump_onkeys(*(S0 + 3), *(S1 + 3), S0 + 4, S1 + 4);
    xorshift128plus_jump_onkeys(*(S0 + 4), *(S1 + 4), S0 + 5, S1 + 5);
    xorshift128plus_jump_onkeys(*(S0 + 5), *(S1 + 5), S0 + 6, S1 + 6);
    xorshift128plus_jump_onkeys(*(S0 + 6), *(S1 + 6), S0 + 7, S1 + 7);

    part1 = _mm512_loadu_si512((const __m512i *) S0);
    part2 = _mm512_loadu_si512((const __m512i *) S1);
}

/*
 Return a 512-bit random "number"
 */
__m512i avx512_xorshift128plus(__m512i &part1, __m512i &part2) {
    __m512i s1 = part1;
    const __m512i s0 = part2;
    part1 = part2;
    s1 = _mm512_xor_si512(part2, _mm512_slli_epi64(part2, 23));
    part2 = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_xor_si512(s1, s0),_mm512_srli_epi64(s1, 18)),
        _mm512_srli_epi64(s0, 5)
    );
    return _mm512_add_epi64(part2, s0);
}