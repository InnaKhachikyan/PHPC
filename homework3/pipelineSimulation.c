/* This program makes a calculation of the expression:
 * (vector1 + vector2 - vector3) * vector4 in 3 stages, waiting for all the threads responsible
 * for cell addition/subtraction/dot product to finish */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define NUMBER_OF_THREADS 8

int *vector1, *vector2, *vector3, *vector4;
pthread_barrier_t barrier1;
pthread_barrier_t barrier2;
int result;

void calculateExpression(void);
void* pipeline(void* arg);
void* addTwoVectors(void* arg);
void* subtractTwoVectors(void* arg);
void* dotProduct(void* arg);

int main(void) {
	vector1 = (int*)malloc(sizeof(int)*NUMBER_OF_THREADS);
	vector2 = (int*)malloc(sizeof(int)*NUMBER_OF_THREADS);
	vector3 = (int*)malloc(sizeof(int)*NUMBER_OF_THREADS);
	vector4 = (int*)malloc(sizeof(int)*NUMBER_OF_THREADS);
	
	if(vector1 == NULL || vector2 == NULL || vector3 == NULL || vector4 == NULL) {
		printf("Memory allocation for vectors failed!\n");
		return 1;
	}	
	srand(time(NULL));
	
	for(int i = 0; i < NUMBER_OF_THREADS; i++) {
		vector1[i] = rand()%20 + 1;
		vector2[i] = rand()%20 + 1;
		vector3[i] = rand()%20 + 1;
		vector4[i] = rand()%20 + 1;
	}
	result = 0;
	calculateExpression();

	printf("RESULT IS %d\n", result);
	
	free(vector1);
	free(vector2);
	free(vector3);
	free(vector4);

	return 0;
}
// (vector1 + vector2 - vector3) * vector4 expression
void calculateExpression() {
	pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*NUMBER_OF_THREADS);
	if(threads == NULL) {
		printf("Memory allocation for threads failed!\n");
		exit(1);
	}
	pthread_barrier_init(&barrier1,NULL,NUMBER_OF_THREADS);
	for(int i = 0; i < NUMBER_OF_THREADS; i++) {
		int* index = (int*)malloc(sizeof(int));
		if(index == NULL) {
			printf("Memory allocation for index failed!\n");
			exit(1);
		}
		*index = i;
	/*	if(pthread_create((threads+i),NULL, &addTwoVectors, (index))) {
			perror("CREATING THE THREAD FAILED!\n");
		}
	*/
		if(pthread_create((threads+i),NULL,&pipeline,(index))) {
			perror("CREATING THE THREAD FAILED!\n");
		}
	}

	for(int i = 0; i < NUMBER_OF_THREADS; i++) {
		if(pthread_join(*(threads+i),NULL)) {
			perror("THREADS JOIN FAILED!\n");
		}
	}

	/* 
	for(int i = 0; i < NUMBER_OF_THREADS; i++) {
		result += vector4[i];
		printf("**** RESULT %d ****\n",result);
	}
	*/

	free(threads);
}

void* pipeline(void* arg) {
	int* index = (int*)arg;
	
	//stage 1: ADDITION
	vector2[*(index)] += vector1[*(index)];
	pthread_barrier_wait(&barrier1);

	//stag2 2: SUBTRACTION
	vector3[*(index)] = vector2[*(index)] - vector3[*(index)];;
	pthread_barrier_wait(&barrier1);

	//stage 3: MULTIPLYING INDICES
	vector4[*(index)] *= vector3[*(index)];
	pthread_barrier_wait(&barrier1);

	// Last thread calculates the final sum of the dot product
	if(*(index) == NUMBER_OF_THREADS - 1) {

		for(int i = 0; i < NUMBER_OF_THREADS; i++) {
			result += vector4[i];
		}	
	}
}

void* addTwoVectors(void* arg) {
	printf("*** ADD_TWO_VECTORS ***\n");
	int* index = (int*)arg;
	vector2[*(index)] += vector1[*(index)];
	pthread_barrier_wait(&barrier1);
	subtractTwoVectors((void*)index);
	return NULL;
}

void* subtractTwoVectors(void* arg) {
	printf("*** SUBTRACT_TWO_VECTORS ***\n");
	int* index = (int*)arg;
	vector3[*(index)] = vector2[*(index)] - vector3[*(index)];;
	pthread_barrier_wait(&barrier1);
	dotProduct((void*)index);
	return NULL;
}

void* dotProduct(void* arg) {
	printf("*** DOT_PRODUCT ***\n");
	int* index = (int*)arg;
	vector4[*(index)] *= vector3[*(index)];
	pthread_barrier_wait(&barrier1);
	return NULL;
}
