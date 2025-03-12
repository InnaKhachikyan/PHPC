#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int padding, rowSize, pixelsSize;

#pragma pack(push, 1)
struct FileHeader {
	unsigned char signature[2];
	uint32_t fileSize;
	uint16_t reserved1;
	uint16_t reserved2;
	uint32_t pixelOffset;
};

struct InfoHeader {
	uint32_t headerSize;
	int32_t width;
	int32_t height;
	uint16_t planes;
	uint16_t bitsPerPixel;
	uint32_t compression;
	uint32_t imageSize;
	int32_t xPixelsPerMeter;
	int32_t yPixelsPerMeter;
	uint32_t colorsUsed;
	uint32_t importantColors;
};
#pragma pack(pop)

void readHeaders(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader);
void iterativeProcessing(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader);
void simdProcessing(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader);
void outputGreyScale(char *outputFileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader, unsigned char *pixels, int pixelsSize);

int main(int argNum, char *args[]) {

	if(argNum < 2) {
		printf("Argument file not given\n");
		return 1;
	}

	if(sizeof(struct FileHeader) != 14 || sizeof(struct InfoHeader) != 40) {
		printf("STRUCT MISALIGNMENT\n");
	}
	
	struct FileHeader *fileHeader = (struct FileHeader*)malloc(sizeof(struct FileHeader));
	struct InfoHeader *infoHeader = (struct InfoHeader*)malloc(sizeof(struct InfoHeader));
	if(fileHeader == NULL || infoHeader == NULL) {
		printf("Memory allocation failed\n");
		return 1;
	}

	char *fileName = args[1];

	readHeaders(fileName, fileHeader, infoHeader);

	clock_t iterStart = clock();
	iterativeProcessing(fileName, fileHeader, infoHeader);
	clock_t iterEnd = clock();
	double iterTotal = (double)(iterEnd - iterStart)/CLOCKS_PER_SEC;

	clock_t simdStart = clock();
	simdProcessing(fileName, fileHeader, infoHeader);
	clock_t simdEnd = clock();
	double simdTotal = (double)(simdEnd - simdStart)/CLOCKS_PER_SEC;

	printf("ITERATIVE TIME: %f sec\n", iterTotal);
	printf("SIMD TIME: %f sec\n", simdTotal);

	free(fileHeader);
	fileHeader = NULL;
	free(infoHeader);
	infoHeader = NULL;

	return 0;
}

void readHeaders(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader) {
	FILE *file = fopen(fileName, "rb");
	if(file == NULL) {
		perror("Couldn't open the file\n");
		return;
	}

	if(fread(fileHeader, sizeof(struct FileHeader), 1, file) != 1) {
		perror("Couldn't read the file header\n");
		return;
	}

	if(fread(infoHeader, sizeof(struct InfoHeader), 1, file) != 1) {
		perror("Couldn't read the info header\n");
		return;
	}

	if(fileHeader->signature[0] != 'B' || fileHeader->signature[1] != 'M') {
		printf("Not a BMP file\n");
		exit(0);
	}

	fclose(file);
}

void iterativeProcessing(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader) {

	int padding = (4 - (infoHeader->width * 3) %4) %4;
	int rowSize = (infoHeader->width * 3) + padding;
	int pixelsSize = rowSize * abs(infoHeader->height);

	unsigned char *pixels = (unsigned char*)malloc(sizeof(char)*pixelsSize);

	if(pixels == NULL) {
		printf("Memory allocation failed\n");
		return;
	}

	FILE *file = fopen(fileName, "rb");

	if(fseek(file, fileHeader->pixelOffset, SEEK_SET) != 0) {
		printf("Failed to move ot the offset\n");
		return;
	}

	if(fread(pixels, pixelsSize, 1, file) != 1) {
		printf("Failed to read the pixel data\n");
		return;
	}

	unsigned char blue, red, green, grey;
	for(int i = 0; i < abs(infoHeader->height); i++) {
		for(int j = 0; j < infoHeader->width; j++) {
			int index = i * rowSize + j * 3;

			blue = pixels[index];
			green = pixels[index+1];
			red = pixels[index+2];

			grey = 0.114*blue + 0.587*green + 0.299*red;

			pixels[index] = grey;
			pixels[index+1] = grey;
			pixels[index+2] = grey;

		}
	}

	outputGreyScale("output.bmp", fileHeader, infoHeader, pixels, pixelsSize);

	fclose(file);
	free(pixels);
	pixels = NULL;
}

void simdProcessing(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader) {
	int padding = (4 - (infoHeader->width * 3) %4) %4;
	int rowSize = (infoHeader->width * 3) + padding;
	int pixelsSize = rowSize * abs(infoHeader->height);

	unsigned char *pixels = (unsigned char*)malloc(sizeof(char)*pixelsSize);
	
	if(pixels == NULL) {
		printf("Memory allocation failed\n");
		return;
	}

	FILE *file = fopen(fileName, "rb");

	if(fseek(file, fileHeader->pixelOffset, SEEK_SET) != 0) {
		printf("Failed to move ot the offset\n");
		return;
	}

	if(fread(pixels, pixelsSize, 1, file) != 1) {
		printf("Failed to read the pixel data\n");
		return;
	}
	__m256i current;
	__m256 blue_weight = _mm256_set1_ps(0.114f);
    __m256 green_weight = _mm256_set1_ps(0.587f);
    __m256 red_weight = _mm256_set1_ps(0.299f);
	
	int i,j;
	for(i = 0; i < abs(infoHeader->height); i++) {
		for(j = 0; j < infoHeader->width; j += 8) {
			current = _mm256_loadu_si256((__m256i*)&pixels[i * rowSize + j * 3]);

			__m256 blue_f = _mm256_cvtepi32_ps(_mm256_and_si256(current, _mm256_set1_epi32(0xFF)));
			__m256 green_f = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(current, 8), _mm256_set1_epi32(0xFF)));
			__m256 red_f = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(current, 16), _mm256_set1_epi32(0xFF)));

			__m256 grey = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(red_f, red_weight), _mm256_mul_ps(green_f, green_weight)), _mm256_mul_ps(blue_f, blue_weight));
			
			float greyValues[8];
			_mm256_storeu_ps(greyValues, grey);

			for(int k = 0; k < 8; k++) {
				int pixelIndex = i * rowSize + (j + k) * 3;
				pixels[pixelIndex] = greyValues[k];
				pixels[pixelIndex + 1] = greyValues[k];
				pixels[pixelIndex + 2] = greyValues[k];
			}
		}
	}
	
	unsigned char blue, red, green, grey;
	for(; i < abs(infoHeader->height); i++) {
		for(; j < infoHeader->width; j++) {
			int index = i * rowSize + j * 3;

			blue = pixels[index];
			green = pixels[index+1];
			red = pixels[index+2];

			grey = 0.114*blue + 0.587*green + 0.299*red;

			pixels[index] = grey;
			pixels[index+1] = grey;
			pixels[index+2] = grey;

		}
	}

	outputGreyScale("output_simd.bmp", fileHeader, infoHeader, pixels, pixelsSize);

	fclose(file);
    free(pixels);
	pixels = NULL;
}

void outputGreyScale(char *outputFileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader, unsigned char *pixels, int pixelsSize) {
	FILE *outputFile = fopen(outputFileName, "wb");
	if(outputFile == NULL) {
		printf("Opening file failed\n");
		return;
	}

	fwrite(fileHeader, sizeof(struct FileHeader), 1, outputFile);
	fwrite(infoHeader, sizeof(struct InfoHeader), 1, outputFile);
	fwrite(pixels, pixelsSize, 1, outputFile);
	fclose(outputFile);
}
