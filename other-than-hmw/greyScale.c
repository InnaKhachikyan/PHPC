%%writefile greyScale.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#pragma pack(push,1)
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
#pragma pack(pop,1)

void readHeaders(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader);
unsigned char* extractPixels(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader);

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

unsigned char* extractPixels(char *fileName, struct FileHeader *fileHeader, struct InfoHeader *infoHeader) {
	int padding = (4 - (infoHeader->width * 3) %4) %4;
	int rowSize = (infoHeader->width * 3) + padding;
	int pixelsSize = rowSize * abs(infoHeader->height);

	unsigned char *pixels = (unsigned char*)malloc(sizeof(char)*pixelsSize);

	if(pixels == NULL) {
		printf("Memory allocation failed\n");
		return NULL;
	}

	FILE *file = fopen(fileName, "rb");

	if(fseek(file, fileHeader->pixelOffset, SEEK_SET) != 0) {
		printf("Failed to move ot the offset\n");
		return NULL;
	}

	if(fread(pixels, pixelsSize, 1, file) != 1) {
		printf("Failed to read the pixel data\n");
		return NULL;
	}
	return pixels;
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

__global__ void image_processing(unsigned char *pixels, int width, int height, int pixelsSize, int rowSize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
  int totalPixels = abs(height) * width;

	if(index < totalPixels) {
		int row = index/width;
		int col = index%width;
		int pixelIndex = row * rowSize + col * 3;

		unsigned char b = pixels[pixelIndex];
        unsigned char g = pixels[pixelIndex + 1];
	    unsigned char r = pixels[pixelIndex + 2];

		unsigned char grey = 0.114*b +0.587*g + 0.299*r;

		pixels[pixelIndex] = grey;
		pixels[pixelIndex+1] = grey;
		pixels[pixelIndex+2] = grey;
	}
}

void greyScale(char *fileName) {
	if(sizeof(struct FileHeader) != 14 || sizeof(struct InfoHeader) != 40) {
		printf("STRUCT MISALIGNMENT\n");
	}
	
	struct FileHeader *fileHeader = (struct FileHeader*)malloc(sizeof(struct FileHeader));
	struct InfoHeader *infoHeader = (struct InfoHeader*)malloc(sizeof(struct InfoHeader));
	if(fileHeader == NULL || infoHeader == NULL) {
		printf("Memory allocation failed\n");
		return;
	}

	readHeaders(fileName, fileHeader, infoHeader);

	int padding = (4 - (infoHeader->width*3)%4)%4;
	int rowSize = (infoHeader->width *3) + padding;
	int pixelsSize = rowSize * abs(infoHeader->height);
	unsigned char *pixels = (unsigned char*)malloc(sizeof(unsigned char)*pixelsSize);
	if(pixels == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	pixels = extractPixels(fileName, fileHeader, infoHeader);

	unsigned char *d_pxls = NULL;
	cudaError_t err = cudaMalloc((void **)&d_pxls, sizeof(unsigned char)*pixelsSize);
	if(err != cudaSuccess) {
		printf("CUDA memory not allocated\n");
		exit(1);
	}
	err = cudaMemcpy(d_pxls, pixels, sizeof(unsigned char)*pixelsSize, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Host to device copy failed with code %s\n", cudaGetErrorString(err));
		exit(1);
	}

	int numThreads = 256;
	int numBlocks = (pixelsSize + numThreads - 1)/numThreads;

	image_processing<<<numBlocks, numThreads>>>(d_pxls, infoHeader->width, infoHeader->height, pixelsSize, rowSize);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("Device sync failed with code %s\n", cudaGetErrorString(err));
		exit(1);
	}
	err = cudaMemcpy(pixels, d_pxls, sizeof(unsigned char)*pixelsSize, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Device to host copy failed with code %s\n", cudaGetErrorString(err));
		exit(1);
	}
	outputGreyScale("output.bmp", fileHeader, infoHeader, pixels, pixelsSize);
}	

int main() {
	greyScale("inputImage.bmp");
	return 0;
}
