#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define NUMBER_OF_SENSORS 7

struct sensor* sensors;
pthread_barrier_t barrier1, barrier2;
float average;

struct sensor {
	int sensorNumber;
	double recentData;
};

void collectData(struct sensor* sensors);
void* readTemperature(void* arg);
void* countAverage(void* args);

int main(void) {
	sensors = (struct sensor*)malloc(sizeof(struct sensor)*(NUMBER_OF_SENSORS+1));
	if(sensors == NULL) {
		printf("Memory allocation for the sensors failed!\n");
		return 1;
	}
	for(int i = 0; i < NUMBER_OF_SENSORS; i++) {
		sensors[i].sensorNumber = i + 1;
	}

	collectData(sensors);

	free(sensors);
}

void collectData(struct sensor* sensors) {
	pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*(NUMBER_OF_SENSORS+1));
	if(threads == NULL) {
		printf("Memory allocation for threads failed!\n");
		exit(1);
	}
	pthread_barrier_init(&barrier1, NULL, NUMBER_OF_SENSORS+1);
	pthread_barrier_init(&barrier2, NULL, NUMBER_OF_SENSORS+1);

	for(int i = 0; i <= NUMBER_OF_SENSORS; i++) {
		if(i == NUMBER_OF_SENSORS) {
			if(pthread_create((threads+NUMBER_OF_SENSORS),NULL,&countAverage, sensors)) {
				perror("Thread creating failed!\n");
			}
		}
		else {
			if(pthread_create((threads+i), NULL, &readTemperature, (sensors + i))) {
				perror("Thread creating failed!\n");
			}
		}
	}

	for(int i = 0; i <= NUMBER_OF_SENSORS; i++) {
		if(pthread_join(*(threads),NULL)) {
			perror("Thread join failed!\n");
		}
	}
	free(threads);
}

void* readTemperature(void* arg) {
	while(1) {
		struct sensor* currentSensor = (struct sensor*) arg;
		currentSensor->recentData = (rand()%71)-25; // number in range from -25 to +45
		printf(" THREAD NUMBER %lu WAITING \n",(unsigned long)pthread_self());
		pthread_barrier_wait(&barrier1);
		pthread_barrier_wait(&barrier2);
		sleep(3); // sleep is added just for better visibility on the terminal
	}
}

void* countAverage(void* args) {
	while(1) {
		average = 0;
		pthread_barrier_wait(&barrier1);
		printf("*** CALCULATING AVERAGE ***\n");
		struct sensor* sensors = (struct sensor*)args;
		int sum = 0;
		for(int i = 0; i < NUMBER_OF_SENSORS; i++) {
			sum += sensors[i].recentData;
		}
		average = sum/NUMBER_OF_SENSORS;
		printf("^^^ AVERAGE DATA IS %f ^^^\n",average);
		pthread_barrier_wait(&barrier2);
	}
}
