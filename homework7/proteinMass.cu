%%writefile proteinMass.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10000000

char *alphabet, *data, *d_data;
double *output, *d_output;
float *protein, *d_protein;

char* initializeArray(char *data, char *alphabet, int size) {
	data = (char*)malloc(sizeof(char)*size);
	if(!data) {
		printf("Memory not allocated\n");
		return NULL;
	}
	srand(time(NULL));
	for(int i = 0; i < SIZE; i++) {
		data[i] = alphabet[rand()%20];
	}
	return data;
}

void cleanup() {
	if(alphabet) {
		free(alphabet);
		alphabet = NULL;
	}
	if(data) {
		free(data);
		data = NULL;
	}
	if(d_data) {
		cudaFree(d_data);
		d_data = NULL;
	}
	if(output) {
		free(output);
		output = NULL;
	}
	if(d_output) {
		cudaFree(d_output);
		d_output = NULL;
	}
	if(protein) {
		free(protein);
		protein = NULL;
	}
	if(d_protein) {
		cudaFree(d_protein);
		d_protein = NULL;
	}
}


char* initializeAlphabet(char *alphabet) {
	alphabet = (char*)malloc(sizeof(char)*26);
	if(!alphabet) {
		printf("Alphabet memory not allocated\n");
		exit(1);
	}
	alphabet[0] = 'A';
	alphabet[1] = 'C';
	alphabet[2] = 'D';
	alphabet[3] = 'E';
	alphabet[4] = 'F';
	alphabet[5] = 'G';
	alphabet[6] = 'H';
	alphabet[7] = 'I';
	alphabet[8] = 'K';
	alphabet[9] = 'L';
	alphabet[10] = 'M';
	alphabet[11] = 'N';
	alphabet[12] = 'P';
	alphabet[13] = 'Q';
	alphabet[14] = 'R';
	alphabet[15] = 'S';
	alphabet[16] = 'T';
	alphabet[17] = 'V';
	alphabet[18] = 'W';
	alphabet[19] = 'Y';
	return alphabet;
}

float* initializeProtein(float *protein) {
	protein = (float *)malloc(sizeof(float) * 26);
	if(!protein) {
		printf("memory for protein mass not allocated\n");
		exit(1);
	}
	protein[0] = 71.03711f;
	protein[1] = 0.0f; 
	protein[2] = 103.00919f;
	protein[3] = 115.02694f;
	protein[4] = 129.04259f;
	protein[5] = 147.06841f;
	protein[6] = 57.02146f;
	protein[7] = 137.05891f;
	protein[8] = 113.08406f;
	protein[9] = 0.0f; 
	protein[10] = 128.09496f;
	protein[11] = 113.08406f;
	protein[12] = 131.04049f;
	protein[13] = 114.04293f;
	protein[14] = 0.0f; 
	protein[15] = 97.05276f;
	protein[16] = 128.05858f;
	protein[17] = 156.10111f;
	protein[18] = 87.03203f;
	protein[19] = 101.04768f;
	protein[20] = 0.0f;     
	protein[21] = 99.06841f; 
	protein[22] = 186.07931f; 
	protein[23] = 0.0f;      
	protein[24] = 163.06333f;  
	protein[25] = 0.0f; 

	return protein;
}

void memoryAllocAndCopy() {
	protein = initializeProtein(protein);
	alphabet = initializeAlphabet(alphabet);

	data = initializeArray(data, alphabet, SIZE);
	if(!data) {
		printf("Initialization failed\n");
		cleanup();
		exit(1);
	}

	output = (double*)malloc(sizeof(double));
	if(!output) {
		printf("Memory allocation Failed\n");
		cleanup();
		exit(1);
	}
	*output = 0;

	cudaError_t err = cudaMalloc((void**)&d_data, sizeof(char)*SIZE);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}
	err = cudaMemcpy(d_data, data, sizeof(char)*SIZE, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}       

	err = cudaMalloc((void**)&d_output, sizeof(double));
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}       
	err = cudaMemcpy(d_output, output, sizeof(double), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}

	err = cudaMalloc((void**)&d_protein, sizeof(float)*26);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}
	err = cudaMemcpy(d_protein, protein, sizeof(float)*26, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}
}

__global__ void protein_mass(char *data, double *output, int n, float *protein) {
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
	__shared__ double partial[256];

	double sum = 0.0;

	if (idx < n) {
		sum += protein[data[idx] - 'A'];
	}

	if (idx + blockDim.x < n) {
		sum += protein[data[idx + blockDim.x] - 'A'];
	}

	if (idx + 2 * blockDim.x < n) {
		sum += protein[data[idx + 2 * blockDim.x] - 'A'];
	}

	if (idx + 3 * blockDim.x < n) {
		sum += protein[data[idx + 3 * blockDim.x] - 'A'];
	}

	if (idx + 4 * blockDim.x < n) {
		sum += protein[data[idx + 4 * blockDim.x] - 'A'];
	}

	if (idx + 5 * blockDim.x < n) {
		sum += protein[data[idx + 5 * blockDim.x] - 'A'];
	}

	if (idx + 6 * blockDim.x < n) {
		sum += protein[data[idx + 6 * blockDim.x] - 'A'];
	}

	if (idx + 7 * blockDim.x < n) {
		sum += protein[data[idx + 7 * blockDim.x] - 'A'];
	}

	partial[tid] = sum;
	__syncthreads();

	if (blockDim.x>=256 && tid < 128) partial[tid] += partial[tid + 128];
	__syncthreads();

	if (blockDim.x>=128 && tid < 64) partial[tid] += partial[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile double *vsmem = partial;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
	if(tid == 0) {
		atomicAdd(output, partial[0]);
	}
}

int main() {

	memoryAllocAndCopy();

	int numThreadsPerBlock = 256;
	int numBlocks = (SIZE + numThreadsPerBlock  * 8- 1)/(numThreadsPerBlock * 8);

	//warmup
	protein_mass<<<numBlocks, numThreadsPerBlock>>>(d_data, d_output, SIZE, d_protein);
	cudaDeviceSynchronize();
	cudaMemset(d_output, 0, sizeof(double));

	clock_t start, end;
	start = clock();
	protein_mass<<<numBlocks, numThreadsPerBlock>>>(d_data, d_output, SIZE, d_protein);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end-start)*1000.0/CLOCKS_PER_SEC;

	cudaError_t err = cudaMemcpy(output, d_output, sizeof(double), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("Memory allocation failed with code %s\n", cudaGetErrorString(err));
		cleanup();
		exit(1);
	}

	start = clock();
	double cpu_output = 0.0;
	for(int i = 0; i < SIZE; i++) {
		cpu_output += protein[data[i]-65];
	}
	end = clock();
	double cpu_time = (double)(end - start) * 1000.0/CLOCKS_PER_SEC;

	if(*output != cpu_output) {
		printf("WRONG OUTPUT\n");
		printf("%f AND %f\n", *output, cpu_output);
		printf("diff %f\n", cpu_output-(*output));
	}
	else {
		printf("GPU res: %f\nCPU res: %f\n", *output, cpu_output);
	}

	printf("GPU TIME: %f\nCPU TIME: %f\n", gpu_time, cpu_time);
	printf("GPU is %f times faster\n", (cpu_time/gpu_time));

	cleanup();
}
