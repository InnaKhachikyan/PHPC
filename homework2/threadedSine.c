#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

double degreesToRadians(double degree);
double factorial(int n);
double power(double x, int n);
double sine(double x, int n);

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
	double partialResult = 0;
	printf("Start is: %d\n",start);	
	while(start <= end) {
                partialResult += (((start%2 == 0) ? 1 : (-1))/(factorial(2*start+1)))*power(x,(2*start+1));
		start++;
	}
	double* rPointer = (double*)malloc(sizeof(double));
	*rPointer = partialResult;
	return (void*)rPointer;
}

double createThreads(int angle, int n, int numberOfThreads) {
        pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*numberOfThreads);
        struct ThreadArguments* args = (struct ThreadArguments*)malloc(sizeof(struct ThreadArguments)*numberOfThreads);

	int rem = n % numberOfThreads;
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
	double* partialResult, finalResult;
        for(int i = 0; i < numberOfThreads; i++) {
                if(pthread_join(*(threads+i), (void*)(&partialResult))) {
                        perror("Thread joining failed!\n");
                }
		else {
			finalResult += *partialResult;
		}
        }

        free(threads);
        free(args);
	
	return finalResult;
}

int main(void) {
	struct timespec startTime, endTime;
	
	clock_gettime(CLOCK_MONOTONIC, &startTime);

	pthread_t sineThread1, sineThread2, sineThread3, sineThread4;
	
	double result = createThreads(215, 40, 4);
	clock_gettime(CLOCK_MONOTONIC, &endTime);
	double totalTime = (endTime.tv_sec - startTime.tv_sec) + ((endTime.tv_nsec - startTime.tv_nsec) / 1e9);
	printf("Final result is: %f\n",result);
	printf("Execution time is: %f\n", totalTime);
	return 0;
}


