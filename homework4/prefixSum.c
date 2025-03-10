#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 1000000001
int *values, *outputScalar, *outputSimd, *outputSimd2;

void scalarPrefixSum(int input[], int output[]);
void simdPrefixSum(int input[], int output[]);
void simdPrefixSum2(int input[], int output[]);
int main() {
	values = (int*)malloc(sizeof(int)*SIZE);
	outputScalar = (int*)malloc(sizeof(int)*SIZE);
	outputSimd = (int*)malloc(sizeof(int)*SIZE);
	outputSimd2 = (int*)malloc(sizeof(int)*SIZE);
	if(values == NULL || outputScalar == NULL || outputSimd == NULL) {
		printf("Memory allocation failed!\n");
		return 1;
	}

	srand(time(NULL));
	//initializing the array
	clock_t initStart = clock();
	for(int i = 0; i < SIZE; i++) {
		values[i] = rand()%10;
		//printf("values[%d] is: %d\n", i, values[i]);
		outputScalar[i] = 0;
		outputSimd[i] = 0;
		outputSimd2[i] = 0;
	}
	clock_t initEnd = clock();
	double initTotal = (double)(initEnd-initStart)/CLOCKS_PER_SEC;

	clock_t scalarStart = clock();
	scalarPrefixSum(values, outputScalar);
	clock_t scalarEnd = clock();
	double scalarTotal = (double)(scalarEnd - scalarStart)/CLOCKS_PER_SEC;

	clock_t simdStart = clock();
	simdPrefixSum(values, outputSimd);
	clock_t simdEnd = clock();
	double simdTotal = (double)(simdEnd-simdStart)/CLOCKS_PER_SEC;

	clock_t simd2Start = clock();
	simdPrefixSum2(values, outputSimd2);
	clock_t simd2End = clock();
	double simd2Total = (double)(simd2End - simd2Start)/CLOCKS_PER_SEC;

	printf("Array init time: %f sec\n", initTotal);
	printf("Scalar prefix sum time: %f sec\n", scalarTotal);
	printf("Simd prefix sum time: %f sec\n", simdTotal);
	printf("Simd2 prefix sum time: %f sec\n", simd2Total);
	printf("SCALAR OUTPUT: %d\n", outputScalar[SIZE-1]);
	printf("SIMD OUTPUT: %d\n", outputSimd[SIZE-1]);
	printf("SIMD2 OUTPUT: %d\n", outputSimd2[SIZE-1]);

	free(values);
	values = NULL;
	free(outputScalar);
	outputScalar = NULL;
	free(outputSimd);
	outputSimd = NULL;
	return 0;
}

void scalarPrefixSum(int input[], int output[]) {
	output[0] = input[0];
	for(int i = 1; i < SIZE; i++) {
		output[i] = output[i-1] + input[i];
	}
}

//VERSION 1
void simdPrefixSum(int input[], int output[]) {
	output[0] = input[0];	
	for(int i = 1; i < SIZE-(SIZE%8); i += 8) {
		__m256i tmpOutput = _mm256_set1_epi32(output[i-1]);
		__m256i chunk = _mm256_loadu_si256((const __m256i*)&input[i]);
		__m256i sum = chunk;

		__m128i lowHalf = _mm256_extracti128_si256(chunk, 0);
		__m128i highHalf = _mm256_extracti128_si256(chunk, 1);
		for(int j = 0; j < 7; j++) {
			if(j < 4) {
				int leftLast = _mm_extract_epi32(lowHalf, 3);
				lowHalf = _mm_slli_si128(lowHalf, 4);
				highHalf = _mm_slli_si128(highHalf, 4);
				highHalf = _mm_insert_epi32(highHalf, leftLast, 0);
			}
			else {
				highHalf = _mm_slli_si128(highHalf, 4);
			}
			__m256i shifted = _mm256_setr_m128i(lowHalf,highHalf);
			sum = _mm256_add_epi32(sum, shifted);
		}
		sum = _mm256_add_epi32(sum, tmpOutput);
		_mm256_storeu_si256((__m256i*)&output[i],sum);
	}
}

// VERSION 2
void simdPrefixSum2(int input[], int output[]) {
    output[0] = input[0];
    for (int i = 1; i < SIZE - (SIZE % 8); i += 8) {
	    __m256i tmpOutput = _mm256_set1_epi32(output[i-1]);
	    __m256i sum = _mm256_loadu_si256((const __m256i*)&input[i]);
	    
	    __m256i shifted = _mm256_slli_si256(sum, 4);
	    sum = _mm256_add_epi32(shifted, sum);

	    shifted = _mm256_slli_si256(sum, 8);
	    sum = _mm256_add_epi32(shifted, sum);

	    __m128i lowHalf = _mm256_extracti128_si256(sum, 0);
	    __m128i highHalf = _mm256_extracti128_si256(sum, 1);

	    int leftLastPrefixSum = _mm_extract_epi32(lowHalf, 3);
	    lowHalf = _mm_set1_epi32(leftLastPrefixSum);
	    highHalf = _mm_add_epi32(highHalf, lowHalf);
	    sum = _mm256_setr_m128i(lowHalf, highHalf);
	    sum = _mm256_add_epi32(sum, tmpOutput);

	    _mm256_storeu_si256((__m256i*)&output[i], sum);
    }
}

