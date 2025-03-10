#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define SIZE 1000000000

int *values;

int scalarMin(int* arr);
int simdMin(int* arr);

int main(void) {
	values = (int*)malloc(sizeof(int)*SIZE);
	if(values == NULL) {
		printf("Memory allocation failed!\n");
		return 1;
	}

	srand(time(NULL)); // pass seed (without this the generated numbers mainly coincide)
	clock_t arrInitStart = clock();
	// initialize the array
	for(int i = 0; i < SIZE; i++) {
		values[i] = rand();
		//printf("%d\n",values[i]);
	}
	clock_t arrInitEnd = clock();
	double arrInit = (double)(arrInitEnd-arrInitStart)/CLOCKS_PER_SEC;
	printf("Array initialized in %f sec\n",arrInit);
	clock_t start = clock();
	int scalar = scalarMin(values);
	clock_t end = clock();
	double total = (double)(end-start)/CLOCKS_PER_SEC;

	clock_t start2 = clock();
	int simd = simdMin(values);
	clock_t end2 = clock();
	double total2 = (double)(end2-start2)/CLOCKS_PER_SEC;

	printf("SCALAR METHOD TIME: %f\n",total);
	printf("SIMD TIME: %f\n", total2);

	printf("SIMD returned: %d, SCALAR returned %d\n", simd, scalar);
	free(values);
	values = NULL;
	return 0;
}

int scalarMin(int* arr) {
	int min = arr[0];
	for(int i = 1; i < SIZE; i++){
		if(arr[i] < min) {
			min = arr[i];
		}
	}
	return min;
}
int simdMin(int* arr) {
	__m256i minVector = _mm256_loadu_si256((const __m256i*)&arr[0]);
	int i;
	for(i = 8; i < SIZE-(SIZE%8); i+=8) {
		__m256i ar = _mm256_loadu_si256((const __m256i*)&arr[i]);
		minVector = _mm256_min_epi32(ar, minVector);
	}

	int tempArr[8];
	_mm256_storeu_si256((__m256i*)tempArr, minVector);	
	int min = tempArr[0];
	for(int j = 1; j < 8; j++) {
		if(tempArr[j] < min) {
			min = tempArr[j];
		}
	}
	for(; i < SIZE; i++) {
		if(arr[i] < min) {
			min = arr[i];
		}
	}
	return min;
}
