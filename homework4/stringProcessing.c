#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <string.h>

char *input, *outputNaive, *outputSimd;
int textLength;

void toUppercase(char *input, char *output, int length);
void toUpperSimd(char *input, char *output, int length);
void readStringFromFile(char *fileName);
void writeOutputToFile();

int main(int argNum, char *args[]) {

	if(argNum < 2) {
		perror("Not enough arguments\n");
		return 1;
	}

	readStringFromFile(args[1]);

	outputNaive = (char*)malloc(sizeof(char)*textLength);
	outputSimd = (char*)malloc(sizeof(char)*textLength);

	if(outputNaive == NULL || outputSimd == NULL) {
		printf("Memory allocation failed!\n");
		return 1;
	}

	clock_t naiveStart = clock();
	toUppercase(input, outputNaive, textLength);
	clock_t naiveEnd = clock();
	double naiveTotal = (double)(naiveEnd-naiveStart)/CLOCKS_PER_SEC;
//	printf("**** NAIVE OUTPUT ****\n%s\n", outputNaive);

	clock_t simdStart = clock();
	toUpperSimd(input, outputSimd, textLength);
	clock_t simdEnd = clock();
	double simdTotal = (double)(simdEnd-simdStart)/CLOCKS_PER_SEC;
//	printf("**** SIMD OUTPUT: ****\n%s\n", outputSimd);

	writeOutputToFile(outputSimd);

	printf("NAIVE APPROACH TIME: %f sec\n", naiveTotal);
	printf("SIMD APPROACH TIME: %f sec\n", simdTotal);

	free(outputNaive);
	outputNaive = NULL;
	free(outputSimd);
	outputSimd = NULL;
	free(input);
	input = NULL;
	return 0;
}

void toUppercase(char *input, char *output, int length) {
	char current;
	for(int i = 0; i < length; i++) {
		current = input[i];
		if(current > 96 && current < 123) {
			current -= 32;
		}
		output[i] = current;
	}
}

void toUpperSimd(char *input, char *output, int length) {
	int i;
	__m256i smallestLowercase = _mm256_set1_epi8(96);
	__m256i greatestUppercase = _mm256_set1_epi8(123);
	__m256i conversion = _mm256_set1_epi8(32);

	for(i = 0; i < length - (length%32); i += 32) {
		__m256i currentLine = _mm256_loadu_si256((const __m256i*)&input[i]);
		__m256i resultCmp1 = _mm256_cmpgt_epi8(currentLine, smallestLowercase);
		__m256i resultCmp2 = _mm256_cmpgt_epi8(greatestUppercase, currentLine);

		resultCmp1 = _mm256_and_si256(resultCmp1, resultCmp2);
	        resultCmp1 = _mm256_and_si256(resultCmp1, conversion);
		currentLine = _mm256_sub_epi8(currentLine, resultCmp1);	

		_mm256_storeu_si256((__m256i*)&output[i], currentLine);
	}
	char current;
	for(; i < length; i++) {
		current = input[i];
		if(current > 96 && current < 123) {
			current -= 32;
		}
		output[i] = current;
	}
}

void readStringFromFile(char *fileName) {
	FILE *file = fopen(fileName, "r");
	if(file == NULL) {
		perror("Couldn't open the file\n");
		return;
	}
	
	if(fseek(file, 0, SEEK_END) != 0) {
		perror("Didn't find the end of file\n");
		return;
	}

	textLength = ftell(file);
	if(textLength == -1) {
		perror("File is empty\n");
		return;
	}

	rewind(file);

	input = (char*)malloc(sizeof(char)*textLength+1);
	if(input == NULL) {
		perror("Memory allocation failed\n");
		return;
	}

	size_t numberOfBytesRead = fread(input, 1, textLength, file);
	if(numberOfBytesRead != textLength) {
		perror("File reading failed\n");
	}
	input[textLength] = '\0';
	fclose(file);
}

void writeOutputToFile(char *output) {
	FILE *file = fopen("output.txt", "w");
	if(file == NULL) {
		perror("Failed to open a file\n");
		return;
	}
	fprintf(file, "%s", output);
	fclose(file);
}
