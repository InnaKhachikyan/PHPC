%%writefile pointMutation.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

char *data_s, *data_t, *dna, *d_data_s, *d_data_t;
unsigned long long *hamming_distance, *d_res;

char* initializeData(char *data, int size, char *dna) {
	data = (char*)malloc(sizeof(char)*size);
	if(!data) {
		printf("Memory not allocated\n");
		exit(1);
	}
	for(int i = 0; i < size; i++) {
		data[i] = dna[rand()%4];
	}
	return data;
}

void cleanup() {
	free(data_s);
        free(data_t);
        free(hamming_distance);
        data_s = NULL;
        data_t = NULL;
        hamming_distance = NULL;
        cudaFree(d_data_s);
        cudaFree(d_data_t);
        cudaFree(d_res);
        d_data_s = NULL;
        d_data_t = NULL;
        d_res = NULL;
}

void memoryAllocAndCopy(){
	hamming_distance = (unsigned long long*)malloc(sizeof(unsigned long long));
	if(!hamming_distance) {
		printf("Memory allocation failed\n");
		cleanup();
		exit(1);
	}

	cudaError_t err = cudaMalloc((void**)&d_data_s,sizeof(data_s[0])*SIZE);
	cudaError_t err2 = cudaMalloc((void**)&d_data_t, sizeof(data_t[0])*SIZE);
	cudaError_t err3 = cudaMalloc((void**)&d_res, sizeof(unsigned long long));
	if(err != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
		printf("Device memory allocation failed\n");
		cleanup();
		exit(1);
	}

	err = cudaMemcpy(d_data_s, data_s, sizeof(data_s[0])*SIZE, cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(d_data_t, data_t, sizeof(data_t[0])*SIZE, cudaMemcpyHostToDevice);
	err3 - cudaMemcpy(d_res, hamming_distance, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if(err != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                printf("Device memory copy failed\n");
		cleanup();
                exit(1);
        }
}

__global__ void count_hamming_distance(char *data_s, char *data_t, int size, unsigned long long *res) {
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
	__shared__ long partialSum[256];

	long partial = 0;

	if(idx < size) {
		partial += (data_s[idx] != data_t[idx]);
	}

	if(idx + blockDim.x < size) {
		partial += (data_s[idx + blockDim.x] != data_t[idx + blockDim.x]);
	}

	if(idx + 2 * blockDim.x < size) {
		partial += (data_s[idx + 2 * blockDim.x] != data_t[idx + 2 * blockDim.x]);
	}

	if(idx + 3 * blockDim.x < size) {
		partial += (data_s[idx + 3 * blockDim.x] != data_t[idx + 3 * blockDim.x]);
	}

	if(idx + 4 * blockDim.x < size) {
		partial += (data_s[idx + 4 * blockDim.x] != data_t[idx + 4 * blockDim.x]);
	}

	if(idx + 5 * blockDim.x < size) {
		partial += (data_s[idx + 5 * blockDim.x]!= data_t[idx + 5 * blockDim.x]);
	}

	if(idx + 6 * blockDim.x < size) {
		partial += (data_s[idx + 6 * blockDim.x] != data_t[idx + 6 * blockDim.x]);
	}

	if(idx + 7 * blockDim.x < size) {
		partial += (data_s[idx + 7 * blockDim.x] != data_t[idx + 7 * blockDim.x]);
	}

	partialSum[tid] = partial;
	__syncthreads();

	if(blockDim.x >= 256 && tid < 128) partialSum[tid] += partialSum[tid + 128];
	__syncthreads();

	if(blockDim.x >= 128 && tid < 64) partialSum[tid] += partialSum[tid + 64];
	__syncthreads();

	if(tid < 32) {
		volatile long *vsmem = partialSum;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if(tid == 0) {
		atomicAdd(res, (unsigned long long)partialSum[0]);
	}
}


int main() {
	dna = (char*)malloc(sizeof(char)*4);
        dna[0] = 'A';
        dna[1] = 'C';
        dna[2] = 'G';
        dna[3] = 'T';

	srand(time(NULL));
	data_s = initializeData(data_s, SIZE, dna);
	data_t = initializeData(data_t, SIZE, dna);

	memoryAllocAndCopy();
	
	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock * 8 - 1)/(numThreadsPerBlock * 8);

	//warmup
	count_hamming_distance<<<numBlocks, numThreadsPerBlock>>>(d_data_s, d_data_t, SIZE, d_res);
	cudaDeviceSynchronize();
	cudaMemset(d_res, 0, sizeof(long));

	clock_t start, end;
	start = clock();
	count_hamming_distance<<<numBlocks, numThreadsPerBlock>>>(d_data_s, d_data_t, SIZE, d_res);
	cudaDeviceSynchronize();
	end = clock();

	double gpu_time = (double)(end - start)*1000/CLOCKS_PER_SEC;
	cudaError_t err = cudaMemcpy(hamming_distance, d_res, sizeof(long), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Memory copy back to host failed\n");
		cleanup();
		return 1;
	}

	unsigned long long cpu_res = 0;
	start = clock();
	for(int i = 0; i < SIZE; i++) {
		cpu_res += (data_s[i] != data_t[i]);
	}
	end = clock();
	double cpu_time = (double)(end - start)*1000/CLOCKS_PER_SEC;

	if(*hamming_distance != cpu_res) {
		printf("WRONG OUTPUT\n");
		printf("GPU RES: %llu\nCPU RES: %llu\n", *hamming_distance, cpu_res);
	}
	else {
		printf("GPU RES: %llu\nCPU RES: %llu\n", *hamming_distance, cpu_res);
	}

	printf("GPU TIME: %f\nCPU TIME: %f\n", gpu_time, cpu_time);
	printf("GPU is %f times faster\n", (cpu_time/gpu_time));

	cleanup();
}


		
