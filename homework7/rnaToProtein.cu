%%writefile rnaToProtein.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 333333

char *data, *d_data, *output, *d_output, *output_cpu;

char* initializeData(char *data, const char **dna, int size) {
	data = (char*)malloc(sizeof(char)*3*size);
        if(!data) {
                printf("Memory not allocated\n");
                exit(1);
        }
	srand(time(NULL));
	int index;
        for(int i = 0; i < size; i++) {
                index = rand()%64;
		data[3*i] = dna[index][0];
		data[3*i + 1] = dna[index][1];
		data[3*i + 2] = dna[index][2];
        }
        return data;
}

void cleanup() {
	free(data);
	free(output);
	data = NULL;
	output = NULL;
	cudaFree(d_data);
	cudaFree(d_output);
	d_data = NULL;
	d_output = NULL;
}

void memoryAllocAndCopy() {
	output = (char*)(malloc(sizeof(char)*SIZE));
	output_cpu = (char*)malloc(sizeof(char)*SIZE);
	if(!output || !output_cpu) {
		printf("Memory allocation failed\n");
	}
	cudaError_t err = cudaMalloc((void**)&d_data, sizeof(data[0])*SIZE*3);
	cudaError_t err2 = cudaMalloc((void**)&d_output, sizeof(char)*SIZE);
	if(err != cudaSuccess || err2 != cudaSuccess) {
		printf("Memory allocation on device failed\n");
		cleanup();
		exit(1);
	}

	err = cudaMemcpy(d_data, data, sizeof(data[0])*SIZE*3, cudaMemcpyHostToDevice);
	//err2 = cudaMemcpy(d_output, output, sizeof(output[0])*SIZE, cudaMemcpyHostToDevice);
	if(err != cudaSuccess || err2 != cudaSuccess) {
		printf("Memory copy to device failed\n");
		cleanup();
		exit(1);
	}
}

__device__ __forceinline__ int map_char(char c) {
	if(c == 'A') return 0;
	if(c == 'C') return 1;
	if(c == 'G') return 2;
	if(c == 'U') return 3;
	return -1;
}

__device__ __forceinline__ int find_index(char *triplet) {
	int i0 = map_char(triplet[0]);
	int i1 = map_char(triplet[1]);
	int i2 = map_char(triplet[2]);
	int index = (i0<<4) | (i1<<2) | i2;
	return index;
}

__constant__ char table_const[64] = {
	'K','N','K','N',
        'T','T','T','T',
        'R','S','R','S',
        'I','I','M','I',
	'Q','H','Q','H',
        'P','P','P','P',
        'R','R','R','R',
        'L','L','L','L',
        'E','D','E','D',
        'A','A','A','A',
        'G','G','G','G',
        'V','V','V','V',
        '*','Y','*','Y',
        'S','S','S','S',
        '*','C','W','C',
        'L','F','L','F'
};

__global__ void rna_to_protein(char *data, char *output, int size_data, int size_output) {
	
	__shared__ char table[64];
	
	int tid = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	
	if(tid < 64) {
		table[tid] = table_const[tid];
	}
	
	__syncthreads();

	if(idx < size_output) {
		char current[3];
		current[0] = data[3 * idx];
		current[1] = data[3 * idx + 1];
		current[2] = data[3 * idx + 2];
	
		int index = find_index(current);

		output[idx] = table[index];
	}
}

int map(char c) {
        if(c == 'A') return 0;
        if(c == 'C') return 1;
        if(c == 'G') return 2;
        if(c == 'U') return 3;
        return -1;
}

int index(char *triplet) {
        int i0 = map(triplet[0]);
        int i1 = map(triplet[1]);
        int i2 = map(triplet[2]);
        int index = (i0<<4) | (i1<<2) | i2;
        return index;
}

int main() {
	
	static const char *dna[64] = {
	    "AAA","AAC","AAG","AAU",
	    "ACA","ACC","ACG","ACU",
	    "AGA","AGC","AGG","AGU",
	    "AUA","AUC","AUG","AUU",
	    "CAA","CAC","CAG","CAU",
	    "CCA","CCC","CCG","CCU",
	    "CGA","CGC","CGG","CGU",
	    "CUA","CUC","CUG","CUU",
	    "GAA","GAC","GAG","GAU",
	    "GCA","GCC","GCG","GCU",
	    "GGA","GGC","GGG","GGU",
	    "GUA","GUC","GUG","GUU",
	    "UAA","UAC","UAG","UAU",
	    "UCA","UCC","UCG","UCU",
	    "UGA","UGC","UGG","UGU",
	    "UUA","UUC","UUG","UUU"
	};

	static const char table[64] = {
	        'K','N','K','N',
	        'T','T','T','T',
	        'R','S','R','S',
	        'I','I','M','I',
	        'Q','H','Q','H',
	        'P','P','P','P',
	        'R','R','R','R',
	        'L','L','L','L',
	        'E','D','E','D',
	        'A','A','A','A',
	        'G','G','G','G',
	        'V','V','V','V',
	        '*','Y','*','Y',
	        'S','S','S','S',
	        '*','C','W','C',
	        'L','F','L','F'
	};


	data = initializeData(data, dna, SIZE);
	
	memoryAllocAndCopy();

	int numThreads = 256;
	int numBlocks = (SIZE + numThreads - 1)/(numThreads);

	clock_t start, end;
	start = clock();
	rna_to_protein<<<numBlocks, numThreads>>>(d_data, d_output, SIZE*3, SIZE);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end - start)*1000/CLOCKS_PER_SEC;
	
	cudaError_t err = cudaMemcpy(output, d_output, sizeof(char)*SIZE, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Memory cpoy from device failed\n");
		cleanup();
		exit(1);
	}
	start = clock();
	char *current_triplet = (char*)malloc(sizeof(char)*3);
	int idx;
	for(int i = 0; i < SIZE; i++) {
		current_triplet[0] = data[i*3];
		current_triplet[1] = data[i*3+1];
		current_triplet[2] = data[i*3+2];
		idx = index(current_triplet);
		output_cpu[i] = table[idx];
	}
	end = clock();
	double cpu_time = (double)(end - start)*1000/CLOCKS_PER_SEC;

	for(int i = 0; i < SIZE; i++) {
		if(output[i] != output_cpu[i]) {
			printf("Wrong output\n");
		}
	}

	printf("GPU time: %f\nCPU time: %f\n", gpu_time, cpu_time);
	



}	
