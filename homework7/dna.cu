%%writefile dna.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000

char *dna, *data, *d_data;
int *cpu_res, *gpu_res, *d_res;

char* initializeArray(char *data, char *dna, int size) {
	data = (char*)malloc(sizeof(char)*size);
	if(!data) {
		printf("Memory not allocated\n");
		exit(1);
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
	if(d_res) {
		cudaFree(d_res);
		d_res = NULL;
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
	
	gpu_res = (int*)malloc(sizeof(int)*4);
	cpu_res = (int*)malloc(sizeof(int)*20);
	if(!cpu_res || !gpu_res) {
		printf("Memory allocation Failed\n");
		cleanup();
		exit(1);
	}

	memset(cpu_res, 0, 20 * sizeof(int));
	memset(gpu_res, 0, 4 * sizeof(int));

	cudaError_t err = cudaMalloc((void**)&d_data, sizeof(data[0])*SIZE);
	cudaError_t err2 = cudaMalloc((void**)&d_res, sizeof(gpu_res[0])*4);
	if(err != cudaSuccess || err2 != cudaSuccess) {
		printf("Device memory allocation failed\n");
		cleanup();
		exit(1);
	}
	err = cudaMemcpy(d_data, data, sizeof(data[0])*SIZE, cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(d_res, gpu_res, sizeof(gpu_res[0])*4, cudaMemcpyHostToDevice);
	if(err != cudaSuccess || err2 != cudaSuccess) {
                printf("Device memory allocation failed\n");                                  
                cleanup();
                exit(1);
        }
}

__global__ void countDna(const char *data, int n, int *result)
{
    int sumA = 0, sumC = 0, sumG = 0, sumT = 0;

    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += grid_stride)
    {
        char current = data[i];
        sumA += (current == 'A');
        sumC += (current == 'C');
        sumG += (current == 'G');
        sumT += (current == 'T');
    }

    atomicAdd(&result[0], sumA);
    atomicAdd(&result[1], sumC);
    atomicAdd(&result[2], sumG);
    atomicAdd(&result[3], sumT);
}

int main() {

	memoryAllocAndCopy();

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock - 1)/numThreadsPerBlock;

	//warmup
	countDna<<<numBlocks, numThreadsPerBlock>>>(d_data, SIZE, d_res);
	cudaDeviceSynchronize();

	cudaMemset(d_res, 0, 4 * sizeof(int));
	
	clock_t start, end;
	start = clock();
	countDna<<<numBlocks, numThreadsPerBlock>>>(d_data, SIZE, d_res);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;
	
	cudaError_t err = cudaMemcpy(gpu_res, d_res, sizeof(int)*4, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Device to host copy failed\n");
		cleanup();
		exit(1);
	}

	start = clock();
	for(int i = 0; i < SIZE; i++) {
		cpu_res[data[i]-65]++;
	}
	end = clock();
	double cpu_time = (double)(end - start) * 1000.0/CLOCKS_PER_SEC;

	if(gpu_res[0] != cpu_res[0] || gpu_res[1] != cpu_res[2] || gpu_res[2] != cpu_res[6] || gpu_res[3] != cpu_res[19]) {
		printf("WRONG OUTPUT!\n");
	}

	printf("GPU TIME: %f\nCPU TIME: %f\n", gpu_time, cpu_time);
  printf("GPU %f times faster\n",(cpu_time/gpu_time));

	cleanup();
}

