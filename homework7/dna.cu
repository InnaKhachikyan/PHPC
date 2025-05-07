%%writefile dna.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000

char *dna, *data;

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

void allocCudaChar(char **d_arr, char *data, int size) {
        cudaError_t err = cudaMalloc((void**)d_arr, sizeof(data[0])*size);
        if(err != cudaSuccess) {
                printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
                free(data);
                data = NULL;
                exit(1);
        }
}

void allocCudaInt(int **d_arr, int *data, int size) {
        cudaError_t err = cudaMalloc((void**)d_arr, sizeof(data[0])*size);
        if(err != cudaSuccess) {
                printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
                free(data);
                data = NULL;
                exit(1);
        }
}


void cudaHTDcopyChar(char *d_arr, char *data, int size) {
        cudaError_t err = cudaMemcpy(d_arr, data, sizeof(data[0])*size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
                printf("Mem copy failed with code %s\n", cudaGetErrorString(err));
                free(data);
                data = NULL;
                cudaFree(d_arr);
                exit(1);
        }
}

void cudaHTDcopyInt(int *d_arr, int *data, int size) {
        cudaError_t err = cudaMemcpy(d_arr, data, sizeof(data[0])*size, cudaMemcpyHostToDevice);
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

__global__ void countDna(char *data, char *dna, int size, int *result) {
	for(int i = 0; i < (size + blockDim.x -1)/blockDim.x; i++) {
		int index = threadIdx.x + blockDim.x*i;
		if(index < size) {
			int idx = data[index]-65;
			atomicAdd(&result[idx], 1);
		}
	}
	
}


__global__ void countDna2(const char *data, int n, int *result)
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
	dna = (char*)malloc(sizeof(char)*4);
	dna[0] = 'A';
	dna[1] = 'C';
	dna[2] = 'G';
	dna[3] = 'T';

	data = initializeArray(data, dna, SIZE);

	if(!data) {
		printf("Initialization failed\n");
		return -1;
	}
	
	int *result = (int*)malloc(sizeof(int)*20);
	memset(result, 0, 20 * sizeof(int));

	char *d_data = NULL;
	allocCudaChar(&d_data, data, SIZE);
	cudaHTDcopyChar(d_data, data, SIZE);

	char *d_dna = NULL;
	allocCudaChar(&d_dna, dna, 4);
	cudaHTDcopyChar(d_dna, dna, 4);

	int *d_res = NULL;
	allocCudaInt(&d_res, result, 20);
	cudaHTDcopyInt(d_res, result, 20);

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock - 1)/numThreadsPerBlock;

	clock_t start, end;
	start = clock();
	countDna<<<numBlocks, numThreadsPerBlock>>>(d_data, d_dna, SIZE, d_res);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;
	cudaDTHcopy(result, d_res, 20);

	start = clock();
	countDna2<<<numBlocks, numThreadsPerBlock>>>(d_data, SIZE, d_res);
	cudaDeviceSynchronize();
	end = clock();
	double gpu2_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;

	printf("A count: %d\n", result[0]);
	printf("C count: %d\n", result[2]);
	printf("G count: %d\n", result[6]);
	printf("T count: %d\n", result[19]);

	memset(result, 0, 20 * sizeof(int));
	start = clock();
	for(int i = 0; i < SIZE; i++) {
		result[data[i]-65]++;
	}
	end = clock();
	double cpu_time = (double)(end - start) * 1000.0/CLOCKS_PER_SEC;
	printf("A count: %d\n", result[0]);
        printf("C count: %d\n", result[2]);
        printf("G count: %d\n", result[6]);
        printf("T count: %d\n", result[19]);

	printf("GPU TIME: %f\nGPU2 TIME: %f\nCPU TIME: %f\n", gpu_time, gpu2_time, cpu_time);
}












