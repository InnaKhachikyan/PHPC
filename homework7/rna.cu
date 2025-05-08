%%writefile rna.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000

char *dna, *data, *d_data, *dd_data, *gpu_res, *cpu_res;

char* initializeArray(char *data, char *dna, int size) {
	data = (char*)malloc(sizeof(char)*size);
	if(!data) {
		printf("Memory not allocated\n");
		return NULL;
	}
	srand(time(NULL));
	for(int i = 0; i < SIZE; i++) {
		data[i] = dna[rand()%4];
	}
	return data;
}

void cleanup() {
	if(dna) {
		free(dna);
		dna = NULL;
	}
	if(data) {
		free(data);
		data = NULL;
	}
	if(cpu_res) {
		free(cpu_res);
		cpu_res = NULL;
	}
	if(gpu_res) {
		free(gpu_res);
		gpu_res = NULL;
	}
	if(d_data) {
		cudaFree(d_data);
		d_data = NULL;
	}
}

void memoryAllocAndCopy() {
	dna = (char*)malloc(sizeof(char)*4);
	if(!dna) {
		printf("Memory alloc failed\n");
		exit(1);
	}
	dna[0] = 'A';
	dna[1] = 'C';
	dna[2] = 'G';
	dna[3] = 'T';

	data = initializeArray(data, dna, SIZE);

	gpu_res = (char*)malloc(sizeof(data[0])*SIZE);
	cpu_res = (char*)malloc(sizeof(data[0])*SIZE);
	if(!cpu_res || !gpu_res) {
		printf("Memory allocation Failed\n");
		cleanup();
		exit(1);
	}

	cudaError_t err = cudaMalloc((void**)&d_data, sizeof(data[0])*SIZE);
	cudaError_t err2 = cudaMalloc((void**)&dd_data, sizeof(data[0])*SIZE);
	if(err != cudaSuccess || err2 != cudaSuccess) {
		printf("Device memory allocation failed\n");
		cleanup();
		exit(1);
	}
	err = cudaMemcpy(d_data, data, sizeof(data[0])*SIZE, cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(dd_data, data, sizeof(data[0])*SIZE, cudaMemcpyHostToDevice);
	if(err != cudaSuccess || err2 != cudaSuccess) {
		printf("Device memory allocation failed\n");                                  
		cleanup();
		exit(1);
	}
}


__global__ void rna_grid_stride(char *data, int n) {
	int idx    = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_stride = blockDim.x * gridDim.x;

	for (int i = idx; i < n; i += grid_stride) {
		char current = data[i];
		if(current == 'T') {
			data[i] = 'U';
			continue;
		}
	}
}

int main() {
	memoryAllocAndCopy();

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock - 1)/numThreadsPerBlock;

	//warmup
	rna_grid_stride<<<numBlocks, numThreadsPerBlock>>>(dd_data, SIZE);
	cudaDeviceSynchronize();

	clock_t start, end;
	start = clock();
	rna_grid_stride<<<numBlocks, numThreadsPerBlock>>>(d_data, SIZE);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;

	cudaError_t err = cudaMemcpy(gpu_res, d_data, sizeof(char)*SIZE, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Device to host memory failed\n");
		cleanup();
		exit(1);
	}

	start = clock();
	for(int i = 0; i < SIZE; i++) {
		if(data[i] == 'T') {
			data[i] = 'U';
			continue;
		}
	}
	end = clock();
	double cpu_time = (double)(end - start) * 1000.0/CLOCKS_PER_SEC;

	for(int i = 0; i < SIZE; i++) {
		if(data[i] != gpu_res[i]) {
			printf("Wrong output\n");
			break;
		}
	}

	printf("GPU TIME: %f\nCPU TIME: %f\n", gpu_time, cpu_time);
	printf("GPU is %f times faster\n", (cpu_time/gpu_time));

	cleanup();
}

