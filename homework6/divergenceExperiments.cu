%%writefile divergenceExperiments.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 100000000
#define WARP_SIZE 32

int *data_no_div, *data_div, *data_div_dummy, *data_min_div;

int* initializeArray (int *data, int size) {
	data = (int*)malloc(sizeof(int)*size);
	if(!data) {
		printf("Memory not allocated\n");
		return NULL;
	}
	for(int i = 0; i < size; i++) {
		data[i] = i;
	}
	return data;
}

void allocCuda(int **d_arr, int *data, int size) {
	cudaError_t err = cudaMalloc((void**)d_arr, sizeof(int)*size);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		free(data);
		data = NULL;
		exit(1);
	}
}

void cudaHTDcopy(int *d_arr, int *data, int size) {
	cudaError_t err = cudaMemcpy(d_arr, data, sizeof(int)*size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Mem copy failed with code %s\n", cudaGetErrorString(err));
		free(data);
		data = NULL;
		cudaFree(d_arr);
		exit(1);
	}
}

void cudaDTHcopy(int *data, int *d_arr, int size) {
	cudaError_t err = cudaMemcpy(data, d_arr, sizeof(int)*size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Mem cpy from device to host failed %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(1);
	}
}

__global__ void noDivergenceKernel(int *d_arr, int size) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index < size) {
		d_arr[index] *= 2;
	}
}

void noDivergenceTest(int *data, int size) {

	int *d_arr = nullptr;
	allocCuda(&d_arr, data, size);
	cudaHTDcopy(d_arr, data, size);

	int numThreadsPerBlock = 256;
	int numBlocks = (size + numThreadsPerBlock -1)/numThreadsPerBlock;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	noDivergenceKernel<<<numBlocks, numThreadsPerBlock>>>(d_arr, size);
	cudaEventRecord(stop);

	cudaError_t err = cudaEventSynchronize(stop);
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(1);
	}

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop);
	printf("NON-DIVERGENT KERNEL time: %f ms\n", totalTime);

	cudaDTHcopy(data, d_arr, size);

	cudaFree(d_arr);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void branchDivergenceKernel(int *d_arr, int size) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index < size) {
		if(index%2 == 0) {
			d_arr[index] *= 2;
		}
		else {
			d_arr[index] *= 3;
		}
	}
}

void divergenceTest(int *data, int size) {

	int *d_arr = nullptr;
	allocCuda(&d_arr, data, size);
	cudaHTDcopy(d_arr, data, size);

	int numThreadsPerBlock = 256;
	int numBlocks = (size + numThreadsPerBlock - 1)/numThreadsPerBlock;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	branchDivergenceKernel<<<numBlocks, numThreadsPerBlock>>>(d_arr, size);
	cudaEventRecord(stop);

	cudaError_t err = cudaEventSynchronize(stop);
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(1);
	}

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop);
	printf("DIVERGENT KERNEL time: %f ms\n", totalTime);

	cudaDTHcopy(data, d_arr, size);

	cudaFree(d_arr);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void warpAlignedDivKernel(int *d_arr, int size) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index < size) {
		if((index/WARP_SIZE) % 2 == 0) {
			d_arr[index] *= 2;
		}       
		else {
			d_arr[index] *= 3;
		}       
	}   
}           

void warpAlignedTest(int *data, int size) {
	int *d_arr = nullptr;
	allocCuda(&d_arr, data, size);
	cudaHTDcopy(d_arr, data, size);

	int numThreadsPerBlock = 256;
	int numBlocks = (size + numThreadsPerBlock - 1)/numThreadsPerBlock;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	warpAlignedDivKernel<<<numBlocks, numThreadsPerBlock>>>(d_arr, size);
	cudaEventRecord(stop);

	cudaError_t err = cudaEventSynchronize(stop);
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(1);
	}

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop);
	printf("WARP ALIGNED DIVERGENCE time: %f ms\n", totalTime);

	cudaDTHcopy(data, d_arr, size);

	cudaFree(d_arr);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

int main() {
	data_no_div = initializeArray(data_no_div, SIZE);
	data_div_dummy = initializeArray(data_div_dummy, SIZE);
	data_div = initializeArray(data_div, SIZE);
	data_min_div = initializeArray(data_min_div, SIZE);
  
	printf("DUMMY RUN: ");
	divergenceTest(data_div_dummy, SIZE);

	printf("*** TIME TESTS ***\n");
  
	divergenceTest(data_div, SIZE);
	// for(int i = 0; i < SIZE; i += 333) {
	// 	printf("%d\n", data_div[i]);
	// }

	noDivergenceTest(data_no_div, SIZE);
	// for(int i = 0; i < SIZE; i += 1000) {
	// 	printf("%d\n", data_no_div[i]);
	// }

	warpAlignedTest(data_min_div, SIZE);
	// for(int i = 0; i < SIZE; i += 333) {
	// 	printf("%d\n", data_min_div[i]);
	// }

	free(data_no_div);
	data_no_div = NULL;
	free(data_div);
	data_div = NULL;
	free(data_min_div);
	data_min_div = NULL;
	return 0;
}

