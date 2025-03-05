#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000000

int *input, *outputScalar, *outputSimd, *weight;

int* scalarConvolution(int *input, int *output, int *weight);
int* simdConvolution(int *input, int *output, int *weight);

int main(void) {
	
	input = (int*)malloc(sizeof(int)*SIZE);
	outputScalar = (int*)malloc(sizeof(int)*SIZE);
	outputSimd = (int*)malloc(sizeof(int)*SIZE);
	weight = (int*)malloc(sizeof(int)*3);
	*(weight) = 2;
	*(weight+1) = 3;
	*(weight+2) = 4;

	srand(time(NULL));
	printf("Init array\n");
	// initialize array
	clock_t initStart = clock();
	for(int i = 0; i < SIZE; i++) {
		input[i] = rand() % 1000;
	}	
	clock_t initEnd = clock();
	printf("Aray init time is: %f sec\n", (double)(initEnd-initStart)/CLOCKS_PER_SEC);

	clock_t scalarStart = clock();
	scalarConvolution(input, outputScalar, weight);
        clock_t scalarEnd = clock();
	printf("SCALAR CONVOLUTION TIME: %f sec\n", (double)(scalarEnd-scalarStart)/CLOCKS_PER_SEC);

	clock_t simdStart = clock();
	simdConvolution(input, outputSimd, weight);
        clock_t simdEnd = clock();
	printf("SIMD CONVOLUTION TIME: %f sec\n", (double)(simdEnd-simdStart)/CLOCKS_PER_SEC);

	free(input);
	input = NULL;
	free(outputScalar);
	outputScalar = NULL;
	free(outputSimd);
	outputSimd = NULL;
	free(weight);
	weight = NULL;
	return 0;
}

int* scalarConvolution(int input[], int output[], int weight[]) {
	for(int i = 1; i < SIZE; i++) {
		output[i] = input[i-1]*weight[0] + input[i]*weight[1] + input[i+1]*weight[2];
	}
	return output;
}

int* simdConvolution(int input[], int output[], int weight[]) {
	__m256i result;
	__m256i weight1 = _mm256_set1_epi32(weight[0]);
	__m256i weight2 = _mm256_set1_epi32(weight[1]);
	__m256i weight3 = _mm256_set1_epi32(weight[2]);

	int i;
	for(i = 1; i < SIZE-(SIZE%8); i += 8) {
		__m256i first = _mm256_loadu_si256((const __m256i*)&input[i-1]);
		__m256i second = _mm256_loadu_si256((const __m256i*)&input[i]);
		__m256i third = _mm256_loadu_si256((const __m256i*)&input[i+1]);

		__m256i product1 = _mm256_mullo_epi32(first, weight1);
		__m256i product2 = _mm256_mullo_epi32(second, weight2);
		__m256i product3 = _mm256_mullo_epi32(third, weight3);

		__m256i sum = _mm256_add_epi32(product1, _mm256_add_epi32(product2, product3));  
		
		_mm256_storeu_si256((__m256i*)&output[i],sum);
	}
	return output;

}
