%%writefile rna.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000

char *dna, *data, *output;

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

void cudaDTHcopy(char *data, char *d_arr, int size) {
        cudaError_t err = cudaMemcpy(data, d_arr, sizeof(char)*size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
                printf("Mem cpy from device to host failed %s\n", cudaGetErrorString(err));
                cudaFree(d_arr);
                free(data);
                data = NULL;
                exit(1);
        }
}

__global__ void rna_grid_stride(char *data, char *output, int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += grid_stride)
    {
        char current = data[i];
	if(current == 'T') {
		data[i] = 'U';
		continue;
	}
    }
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
	
	output = (char*)malloc(sizeof(char)*SIZE);
	if(!output) {
		printf("Memory for output not allocated\n");
		return -1;
	}

	char *d_data = NULL;
	allocCudaChar(&d_data, data, SIZE);
	cudaHTDcopyChar(d_data, data, SIZE);

	char *d_output = NULL;
	allocCudaChar(&d_output, output, SIZE);
	cudaHTDcopyChar(d_output, output, SIZE);

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock - 1)/numThreadsPerBlock;

	printf("WARMUP\n");
	rna_grid_stride<<<numBlocks, numThreadsPerBlock>>>(d_data, d_output, SIZE);
	cudaDeviceSynchronize();

	printf("SECOND Kernel CALL\n");
	clock_t start, end;
	start = clock();
	rna_grid_stride<<<numBlocks, numThreadsPerBlock>>>(d_data, d_output, SIZE);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;
	cudaDTHcopy(output, d_data, SIZE);

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
		if(data[i] != output[i]) {
			printf("Wrong output\n");
      break;
		}
	}

	printf("GPU TIME: %f\nCPU TIME: %f\n", gpu_time, cpu_time);
}

