#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000000

int *input, *outputNaive, *outputSimd, *outputSimd2, *weight;

void naiveConvolution(int *input, int *output, int *weight);
void simdConvolution(int *input, int *output, int *weight);
void simdConvolution2(int *input, int *output, int *weight);

int main(void) {
	
	input = (int*)malloc(sizeof(int)*SIZE);
	outputNaive = (int*)malloc(sizeof(int)*SIZE);
	outputSimd = (int*)malloc(sizeof(int)*SIZE);
	outputSimd2 = (int*)malloc(sizeof(int)*SIZE);
	weight = (int*)malloc(sizeof(int)*3);
	if(input == NULL || outputNaive == NULL || outputSimd == NULL || outputSimd2 == NULL || weight == NULL) {
		printf("Memory allocation failed!\n");
		return 1;
	}

	*(weight) = 2;
	*(weight+1) = 3;
	*(weight+2) = 4;

	srand(time(NULL));
	printf("Init array\n");

	clock_t initStart = clock();
	for(int i = 0; i < SIZE; i++) {
		input[i] = rand() % 1000;
	}	
	clock_t initEnd = clock();
	printf("Aray init time is: %f sec\n", (double)(initEnd-initStart)/CLOCKS_PER_SEC);

	clock_t naiveStart = clock();
	naiveConvolution(input, outputNaive, weight);
    clock_t naiveEnd = clock();
	printf("NAIVE APPROACH CONVOLUTION TIME: %f sec\n", (double)(naiveEnd-naiveStart)/CLOCKS_PER_SEC);

	clock_t simdStart = clock();
	simdConvolution(input, outputSimd, weight);
    clock_t simdEnd = clock();
	printf("SIMD CONVOLUTION TIME: %f sec\n", (double)(simdEnd-simdStart)/CLOCKS_PER_SEC);

	clock_t simd2Start = clock();
	simdConvolution2(input, outputSimd2, weight);
    clock_t simd2End = clock();
	printf("SIMD CONVOLUTION2 TIME: %f sec\n", (double)(simd2End-simd2Start)/CLOCKS_PER_SEC);


	printf("NAIVE OUTPUT[SIZE-1]: %d\n", outputNaive[SIZE-1]);
	printf("SIMD OUTPUT[SIZE-1]: %d\n", outputSimd[SIZE-1]);
	printf("SIMD2 OUTPUT[SIZE-1]: %d\n", outputSimd2[SIZE-1]);

	free(input);
	input = NULL;
	free(outputNaive);
	outputNaive = NULL;
	free(outputSimd);
	outputSimd = NULL;
	free(outputSimd2);
	outputSimd2 = NULL;
	free(weight);
	weight = NULL;
	return 0;
}

void naiveConvolution(int input[], int output[], int weight[]) {
	for(int i = 1; i < SIZE; i++) {
		output[i] = input[i-1]*weight[0] + input[i]*weight[1] + input[i+1]*weight[2];
	}
}
//VERSION 1
void simdConvolution(int input[], int output[], int weight[]) {
	__m256i weight0 = _mm256_set1_epi32(weight[0]);
	__m256i weight1 = _mm256_set1_epi32(weight[1]);
	__m256i weight2 = _mm256_set1_epi32(weight[2]);

	int i;
	for(i = 1; i < SIZE-(SIZE%8); i += 8) {
		__m256i first = _mm256_loadu_si256((const __m256i*)&input[i-1]);
		__m256i second = _mm256_loadu_si256((const __m256i*)&input[i]);
		__m256i third = _mm256_loadu_si256((const __m256i*)&input[i+1]);

		__m256i product1 = _mm256_mullo_epi32(first, weight0);
		__m256i product2 = _mm256_mullo_epi32(second, weight1);
		__m256i product3 = _mm256_mullo_epi32(third, weight2);

		__m256i sum = _mm256_add_epi32(product1, _mm256_add_epi32(product2, product3));  
		
		_mm256_storeu_si256((__m256i*)&output[i],sum);
	}
	for(; i < SIZE; i++) {
		output[i] = input[i-1]*weight[0] + input[i]*weight[1] + input[i+1]*weight[2];
	}
}

//VERSION 2
void simdConvolution2(int input[], int output[], int weight[]) {
	__m256i weight0 = _mm256_set1_epi32(weight[0]);
	__m256i weight1 = _mm256_set1_epi32(weight[1]);
	__m256i weight2 = _mm256_set1_epi32(weight[2]);

	int i;
	for(i = 1; i < SIZE-(SIZE%6); i += 6) {
		__m256i currentLoad = _mm256_loadu_si256((const __m256i*)&input[i-1]);
		__m128i leftHalf = _mm256_extracti128_si256(currentLoad, 0);
		__m128i rightHalf = _mm256_extracti128_si256(currentLoad, 1);

		int rightFirst = _mm_extract_epi32(rightHalf,0);
		leftHalf = _mm_srli_si128(leftHalf,4);
		rightHalf = _mm_srli_si128(rightHalf,4);
		leftHalf = _mm_insert_epi32(leftHalf,rightFirst,3);

		__m256i firstShift = _mm256_setr_m128i(leftHalf,rightHalf);
		
		rightFirst = _mm_extract_epi32(rightHalf,0);
		leftHalf = _mm_srli_si128(leftHalf,4);
		rightHalf = _mm_srli_si128(rightHalf,4);
		leftHalf = _mm_insert_epi32(leftHalf,rightFirst,3);

		__m256i secondShift = _mm256_setr_m128i(leftHalf,rightHalf);

		__m256i product1 = _mm256_mullo_epi32(currentLoad, weight0);
		__m256i product2 = _mm256_mullo_epi32(firstShift, weight1);
		__m256i product3 = _mm256_mullo_epi32(secondShift, weight2);

		__m256i sum = _mm256_add_epi32(product1, _mm256_add_epi32(product2, product3));  
		
		int temp[8];
		_mm256_storeu_si256((__m256i*)&temp[0],sum);

		for (int j = 0; j < 6; j++) {
        	output[i + j] = temp[j];
    	}
	}
	for(; i < SIZE; i++) {
		output[i] = input[i-1]*weight[0] + input[i]*weight[1] + input[i+1]*weight[2];
	}
}
