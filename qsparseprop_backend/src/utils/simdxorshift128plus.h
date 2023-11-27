/**
 * AVX implementation of XORshift. This will be used to generate random numbers.
 *
 * Initial version based on Sebastiano Vigna (vigna@acm.org)
 * http://xorshift.di.unimi.it/xorshift128plus.c
 *
 * AVX implementation done by Daniel Lemire (https://github.com/lemire)
 * https://github.com/lemire/SIMDxorshift
 *
 *
 */
#ifndef QSPARSEPROP_SIMDXORSHIFT128PLUS_H
#define QSPARSEPROP_SIMDXORSHIFT128PLUS_H

#include <immintrin.h>
#include <cstdint>

/**
* You can create a new key like so...
*  __m512i part1;
*  __m512i part2;
*  avx_xorshift128plus_init(324,4444,&part1,&part2);
*
* This feeds the two integers (324 and 4444) as seeds to the random
* number generator.
*
*  Then you can generate random numbers like so...
*      avx_xorshift128plus(&part1,&part2);
* If your application is threaded, each thread should have its own
* key.
*
*
* The seeds (key1 and key2) should be non-zero. You are responsible for
* checking that they are non-zero.
*/
void avx_xorshift128plus_init(uint64_t key1, uint64_t key2, __m256i &part1, __m256i &part2);

/*
Return a 256-bit random "number"
*/
__m256i avx_xorshift128plus(__m256i &part1, __m256i &part2);

/**
* You can create a new key like so...
*  __m512i part1;
*  __m512i part2;
*  avx_xorshift128plus_init(324,4444,&part1,&part2);
*
* This feeds the two integers (324 and 4444) as seeds to the random
* number generator.
*
*  Then you can generate random numbers like so...
*      avx_xorshift128plus(&part1, &part2);
* If your application is threaded, each thread should have its own
* key.
*
*
* The seeds (key1 and key2) should be non-zero. You are responsible for
* checking that they are non-zero.
*/
void avx512_xorshift128plus_init(uint64_t key1, uint64_t key2, __m512i &part1, __m512i &part2);

/**
* Return a 256-bit random "number"
*/
__m512i avx512_xorshift128plus(__m512i &part1, __m512i &part2);

#endif //QSPARSEPROP_SIMDXORSHIFT128PLUS_H
