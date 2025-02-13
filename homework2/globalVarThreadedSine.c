#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

double result;

double degreesToRadians(double degree);
double factorial(int n);
double power(double x, int n);
double sine(double x, int n);
void createThreadsForSine(int x, int n, int numberOfThreads);

struct ThreadArguments {
	double x;
	int start;
	int end;
};

double degreesToRadians(double degree) {
        return degree * (PI/180.0);
}

double factorial(int n) {
        double factorial = 1;
        for(int i = 1; i <= n; i++) {
                factorial = factorial * i;
        }
        return factorial;
}

double power(double x, int n) {
        if(n == 0) {
                return 1;
        }
        double partial = power(x, (int)n/2);
        double result = partial * partial;
        if(n%2 == 1) {
                result *= x;
        }
        return result;
}

double sine(double x, int n) {
        double result = 0.0;
        x = degreesToRadians(x);

        for(int i = 0; i <= n; i++) {

                result += (((i%2 == 0) ? 1 : (-1))/(factorial(2*i+1)))*power(x,(2*i+1));
        }
        return result;
}

void* partialSine(void* arg) {
	struct ThreadArguments* args = (struct ThreadArguments*)arg;
	double x = degreesToRadians(args->x);
	int start = args->start;
	int end = args->end;
	printf("Start is: %d\n",start);
	printf("End is: %d\n", end);
	//printf("Result is: %f\n", result);	
	while(start <= end) {
                result += (((start%2 == 0) ? 1 : (-1))/(factorial(2*start+1)))*power(x,(2*start+1));
		start++;
	}
	return NULL;
}

void createThreads(int angle, int n, int numberOfThreads) {
	pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*numberOfThreads);
	struct ThreadArguments* args = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments)*numberOfThreads);
	int rem = n%numberOfThreads;
	int size = (int)n/numberOfThreads;
	for(int i = 0; i < numberOfThreads; i++) {
		struct ThreadArguments* current = args + i;
		current->x = angle;
		current->start = size*i;
		current->end = size*(i+1);
		
		if(i == (numberOfThreads-1)) {
			current->end += rem;
		}
	}	

	for(int i = 0; i < numberOfThreads; i++) {
		if(pthread_create((threads+i),NULL, &partialSine, (args+i))) {
			perror("Thread Creating failed!\n");
		}
	}
	for(int i = 0; i < numberOfThreads; i++) {
		if(pthread_join(*(threads+i), NULL)) {
			perror("Thread joining failed!\n");
		}
	}
	free(threads);
	free(args);
}



int main(void) {
	struct timespec startTime, endTime;
	
	clock_gettime(CLOCK_MONOTONIC, &startTime);
	
	/*
	pthread_t sineThread1, sineThread2, sineThread3, sineThread4;
	
	struct ThreadArguments* args1 = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments));
	args1->x = 215;
	args1->start = 0;
	args1->end = 10;

	struct ThreadArguments* args2 = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments));
	args2->x = 215;
	args2->start = 11;
	args2->end = 20;

	struct ThreadArguments* args3 = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments));
	args3->x = 215;
	args3->start = 21;
	args3->end = 30;

	struct ThreadArguments* args4 = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments));
	args4->x = 215;
	args4->start = 31;
	args4->end = 40;

	if(pthread_create(&sineThread1, NULL, &partialSine, (void*)args1)) {
		printf("Threading 1 failed\n");
		return -1;
	}
	
	if(pthread_create(&sineThread2, NULL, &partialSine, (void*)args2)) {
		printf("Threading 2 failed\n");
		return -1;
	}

	if(pthread_create(&sineThread3, NULL, &partialSine, (void*)args3)) {
		printf("Threading 3 failed\n");
		return -1;
	}

	if(pthread_create(&sineThread4, NULL, &partialSine, (void*)args4)) {
		printf("Threading 4 failed\n");
		return -1;
	}

	if(pthread_join(sineThread1,NULL)) {
		printf("Join 1 failed!\n");
		return -1;
	}
	
	if(pthread_join(sineThread2, NULL)) {
		printf("Join 2 failed!\n");
		return -1;
	}

	if(pthread_join(sineThread3, (void*) NULL)) {
		printf("Join 3 failed!\n");
		return -1;
	}
	
	if(pthread_join(sineThread4, NULL)) {
		printf("Join 4 failed!\n");
		return -1;
	} 
	*/
	createThreads(215, 40, 4);

	clock_gettime(CLOCK_MONOTONIC, &endTime);
	double totalTime = (endTime.tv_sec - startTime.tv_sec) + ((endTime.tv_nsec - startTime.tv_nsec) / 1e9);

	printf("Final result is: %f\n", result);
	printf("Execution time is: %f\n", totalTime);

	/* free(args1);
	free(args2);
	free(args3);
	free(args4); */
	
	return 0;
}
 
