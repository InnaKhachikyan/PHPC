#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 1000000001
int *values, *outputScalar, *outputSimd;

void scalarPrefixSum(int input[], int output[]);
void simdPrefixSum(int input[], int output[]);

int main() {
	values = (int*)malloc(sizeof(int)*SIZE);
	outputScalar = (int*)malloc(sizeof(int)*SIZE);
	outputSimd = (int*)malloc(sizeof(int)*SIZE);
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

	printf("Array init time: %f sec\n", initTotal);
	printf("Scalar prefix sum time: %f sec\n", scalarTotal);
	printf("Simd prefix sum time: %f sec\n", simdTotal);

	printf("SCALAR OUTPUT: %d\n", outputScalar[SIZE-1]);
	printf("SIMD OUTPUT: %d\n", outputSimd[SIZE-1]);

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

void simdPrefixSum(int input[], int output[]) {
	output[0] = input[0];	
	for(int i = 1; i < SIZE-(SIZE%8); i += 8) {
		__m256i tmpOutput = _mm256_set1_epi32(output[i-1]);
		__m256i chunk = _mm256_loadu_si256((const __m256i*)&input[i]);

		__m128i lowHalf = _mm256_extracti128_si256(chunk, 0);
		__m128i highHalf = _mm256_extracti128_si256(chunk, 1);
		
		int leftLast = _mm_extract_epi32(lowHalf, 3);
		lowHalf = _mm_slli_si128(lowHalf, 4);
		highHalf = _mm_slli_si128(highHalf, 4);
		highHalf = _mm_insert_epi32(highHalf, leftLast, 0);

		__m256i one = _mm256_setr_m128i(lowHalf, highHalf);

		leftLast = _mm_extract_epi32(lowHalf, 3);
		lowHalf = _mm_slli_si128(lowHalf, 4);
		highHalf = _mm_slli_si128(highHalf, 4);
		highHalf = _mm_insert_epi32(highHalf, leftLast, 0);

		__m256i two = _mm256_setr_m128i(lowHalf, highHalf);

		leftLast = _mm_extract_epi32(lowHalf, 3);
		lowHalf = _mm_slli_si128(lowHalf, 4);
		highHalf = _mm_slli_si128(highHalf, 4);
		highHalf = _mm_insert_epi32(highHalf, leftLast, 0);

		__m256i three = _mm256_setr_m128i(lowHalf, highHalf);

		leftLast = _mm_extract_epi32(lowHalf, 3);
		lowHalf = _mm_slli_si128(lowHalf, 4);
		highHalf = _mm_slli_si128(highHalf, 4);
		highHalf = _mm_insert_epi32(highHalf, leftLast, 0);

		__m256i four = _mm256_setr_m128i(lowHalf, highHalf);

		highHalf = _mm_slli_si128(highHalf, 4);

		__m256i five = _mm256_setr_m128i(lowHalf, highHalf);

		highHalf = _mm_slli_si128(highHalf, 4);

		__m256i six = _mm256_setr_m128i(lowHalf, highHalf);

		highHalf = _mm_slli_si128(highHalf, 4);

		__m256i seven = _mm256_setr_m128i(lowHalf, highHalf);

		__m256i sum1 = _mm256_add_epi32(chunk, one);
		__m256i sum2 = _mm256_add_epi32(sum1, two);
		__m256i sum3 = _mm256_add_epi32(sum2, three);
		__m256i sum4 = _mm256_add_epi32(sum3, four);
		__m256i sum5 = _mm256_add_epi32(sum4, five);
		__m256i sum6 = _mm256_add_epi32(sum5, six);
		__m256i sum7 = _mm256_add_epi32(sum6, seven);
		__m256i sum8 = _mm256_add_epi32(sum7, tmpOutput);
		_mm256_storeu_si256((__m256i*)&output[i],sum8);

	}
}

