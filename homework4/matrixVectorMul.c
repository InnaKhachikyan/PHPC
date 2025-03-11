#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 12
#define COLS 100000003

int *vector, *outputVector, *simdOutputVector;
int **matrix;

void matrixVectorIter(int **matrix, int vector[], int output[]);
void simdMatrixVector(int **matrix, int vector[], int output[]);

int main() {

	vector = (int*)malloc(sizeof(int)*COLS);
	matrix = (int**)malloc(sizeof(int*)*ROWS);
	outputVector = (int*)malloc(sizeof(int)*ROWS);
	simdOutputVector = (int*)malloc(sizeof(int)*ROWS);

	if(vector == NULL || matrix == NULL || outputVector == NULL || simdOutputVector == NULL) {
		printf("Memory allocation failed\n");
		return 1;
	}
	srand(time(NULL));
	printf("INIT ARRAY\n");
	clock_t initStart = clock();
	for(int i = 0; i < ROWS; i++) {
		outputVector[i] = 0;
		simdOutputVector[i] = 0;
		matrix[i] = (int*)malloc(sizeof(int)*COLS);
		for(int j = 0; j < COLS; j++) {
			matrix[i][j] = rand() % 10;
			if(i == 0) {
				vector[j] = rand()%10;
			}
		}
	}
	clock_t initEnd = clock();
	double initTotal = (double)(initEnd-initStart)/CLOCKS_PER_SEC;
	printf("ARRAY INIT TIME: %f sec\n", initTotal);

	clock_t iterStart = clock();
	printf("OUTPUT ITERATIVE\n");
	matrixVectorIter(matrix, vector, outputVector);
	for(int i = 0; i < ROWS; i++) {
		printf("%d ", outputVector[i]);
	}
	printf("\n");
	clock_t iterEnd = clock();
	double iterTotal = (double)(iterEnd - iterStart)/CLOCKS_PER_SEC;
	printf("ITERATIVE TOTAL TIME: %f sec\n",iterTotal); 

	clock_t simdStart = clock();
	printf("SIMD OUTPUT\n");
	simdMatrixVector(matrix, vector, simdOutputVector);
	for(int i = 0; i < ROWS; i++ ) {
		printf("%d ", simdOutputVector[i]);
	}
	printf("\n");
	clock_t simdEnd = clock();

	double simdTotal = (double)(simdEnd - simdStart)/CLOCKS_PER_SEC;
	printf("SIMD TOTAL TIME: %f sec\n",simdTotal); 

	free(matrix);
	matrix = NULL;
	free(vector);
	vector = NULL;
	free(outputVector);
	outputVector = NULL;
	free(simdOutputVector);
	simdOutputVector = NULL;
}

void matrixVectorIter(int **matrix, int vector[], int output[]) {
	for(int i = 0; i < ROWS; i++) {
		for(int j = 0; j < COLS; j++) {
			output[i] += matrix[i][j]*vector[j];
		}
	}
}

void simdMatrixVector(int **matrix, int vector[], int output[]){
	__m256i avxVectors[COLS/8+1];
	int k,i,j;
	for(k = 0; k < COLS/8; k++) {
		avxVectors[k] = _mm256_loadu_si256((const __m256i*)&vector[k*8]);
	}
	for(i = 0; i < ROWS; i++) {
		int sum = 0;
		for(j = 0; j < COLS-(COLS%8); j+=8) {
			__m256i arg1 = _mm256_loadu_si256((const __m256i*)&matrix[i][j]);
			__m256i arg2 = avxVectors[j/8];
			__m256i product = _mm256_mullo_epi32(arg1, arg2);

			// horizontal addition
			__m256i hadd = _mm256_hadd_epi32(product, product);
			hadd = _mm256_hadd_epi32(hadd, hadd);
			int lowerSum = _mm256_extract_epi32(hadd, 3);
			int upperSum = _mm256_extract_epi32(hadd, 7);
			sum += lowerSum + upperSum;
		}
		output[i] = sum;
		for(; j < COLS; j++) {
			output[i] += matrix[i][j]*vector[j];
		}
	}	
}
