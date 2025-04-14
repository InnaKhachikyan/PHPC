%%writefile noDivergenceKernel.cu
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

__global__ void no_divergence(int *data, int numElements) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index < numElements) {
		data[index] *= 2;
	}
}

void runProgram() {
	int *data = (int*)malloc(sizeof(int)*SIZE);
	if(!data) {
		printf("Memory not allocated\n");
		return;
	}
	for(int i = 0; i < SIZE; i++) {
		data[i] = i;
	}

	int *d_arr = nullptr;
	cudaError_t err = cudaMalloc((void**)&d_arr, sizeof(int)*SIZE);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		free(data);
		data = NULL;
		exit(0);
	}
	err = cudaMemcpy(d_arr, data, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Mem copy failed with code %s\n", cudaGetErrorString(err));
		free(data);
		data = NULL;
		cudaFree(d_arr);
		exit(0);
	}

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock -1)/numThreadsPerBlock;
	
	no_divergence<<<numBlocks, numThreadsPerBlock>>>(d_arr, SIZE);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(0);
	}
	
	err = cudaMemcpy(data, d_arr, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Mem cpy from device to host failed %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(0);
	}
	cudaFree(d_arr);

	for(int i = 0; i < SIZE; i++) {
    if(i%100 == 0)
	  	printf("%d\n", data[i]);
	}

	free(data);
	data = NULL;
}

int main() {
	runProgram();
	return 0;
}
