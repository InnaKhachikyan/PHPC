%%writefile divergenceExperiments.cu
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

int *data_no_div, *data_div, *data_min_div;

int* initializeArray (int *data, int size) {
	data = (int*)malloc(sizeof(int)*size);
        if(!data) {
                printf("Memory not allocated\n");
                return NULL;
        }
        for(int i = 0; i < SIZE; i++) {
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
                exit(0);
        }
}

void cudaHTDcopy(int *d_arr, int *data, int size) {
	cudaError_t err = cudaMemcpy(d_arr, data, sizeof(int)*size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
                printf("Mem copy failed with code %s\n", cudaGetErrorString(err));
                free(data);
                data = NULL;
                cudaFree(d_arr);
                exit(0);
        }
}

void cudaDTHcopy(int *data, int *d_arr, int size) {
	cudaError_t err = cudaMemcpy(data, d_arr, sizeof(int)*size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
                printf("Mem cpy from device to host failed %s\n", cudaGetErrorString(err));
                cudaFree(d_arr);
                free(data);
                data = NULL;
                exit(0);
        }
}

__global__ void no_divergence(int *data, int numElements) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index < numElements) {
		data[index] *= 2;
	}
}

void noDivergence(int *data, int size) {

	int *d_arr = nullptr;
	allocCuda(&d_arr, data, size);
	cudaHTDcopy(d_arr, data, size);

	int numThreadsPerBlock = 256;
	int numBlocks = (size + numThreadsPerBlock -1)/numThreadsPerBlock;
	no_divergence<<<numBlocks, numThreadsPerBlock>>>(d_arr, size);

	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(0);
	}

	cudaDTHcopy(data, d_arr, size);

	cudaFree(d_arr);
}

__global__ void div_kernel(int *d_arr, int size) {
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

void divergence(int *data, int size) {

	int *d_arr = nullptr;
	allocCuda(&d_arr, data, size);
	cudaHTDcopy(d_Arr, data, size);

	int numThreadsPerBlock = 256;
	int numBlocks = (size + numThreadsPerBlock - 1)/numThreadsPerBlock;

	div_kernel<<<numBlocks, numThreadsPerBlock>>>(d_arr, size);

	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("Synchronization failed with code %s\n", cudaGetErrorString(err));
		cudaFree(d_arr);
		free(data);
		data = NULL;
		exit(0);
	}

	cudaDTHcopy(data, d_arr, size);

	cudaFree(d_arr);
}

int main() {
	data_no_div = initializeArray(data_no_div, SIZE);
	data_div = initializeArray(data_div, SIZE);
	data_min_div = initializeArray(data_min_div, SIZE);

	noDivergence(data_no_div, SIZE);
	for(int i = 0; i < SIZE; i += 1000) {
	printf("%d\n", data_no_div[i]);
 	}

	divergence(data_div, SIZE);
	for(int i = 0; i < SIZE; i += 333) {
		printf("%d\n", data_div[i]);
	}

	free(data_no_div);
	data_no_div = NULL;
	free(data_div);
	data_div = NULL;
	free(data_min_div);
	data_min_div = NULL;
	return 0;
}

