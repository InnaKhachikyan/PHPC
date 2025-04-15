#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_CONSUMERS 4
#define NUM_PRODUCERS 6
#define BUFFER_SIZE 5

int *buffer;
int producerIndex;  //stores the first available index for the producer,
int consumerIndex;
int producedElements = 0;
pthread_spinlock_t spinlock;

void create_threads();
void* produce(void *arg);
void* consume(void *arg);

int main() {
	buffer = (int*)malloc(sizeof(int)*BUFFER_SIZE);
	if(buffer == NULL) {
		printf("Memory not allocated\n");
		return 0;
	}

	for(int i = 0; i < BUFFER_SIZE; i++) {
		buffer[i] = 0;
	}

	create_threads();

	free(buffer);
	buffer = NULL;
}

void create_threads() {
	pthread_t *producers = (pthread_t*)malloc(sizeof(pthread_t)*NUM_PRODUCERS);
	pthread_t *consumers = (pthread_t*)malloc(sizeof(pthread_t)*NUM_CONSUMERS);
	if(producers == NULL || consumers == NULL) {
		printf("Memory for threads not allocated\n");
		exit(1);
	}

	pthread_spin_init(&spinlock, 0);
	producerIndex = 0;
	consumerIndex = 0;

	for(int i =0; i < NUM_PRODUCERS; i++) {
		if(pthread_create((producers+i), NULL, &produce, NULL)) {
			perror("Producers creation failed\n");
		}
	}

	for(int i = 0; i < NUM_CONSUMERS; i++) {
		if(pthread_create((consumers+i), NULL, &consume, NULL)) {
			perror("Consumers creation failed\n");
		}
	}

	for(int i = 0; i < NUM_PRODUCERS; i++) {
		if(pthread_join(*(producers+i), NULL)) {
			perror("Producers joining failed\n");
		}
	}

	for(int i = 0; i < NUM_CONSUMERS; i++) {
		if(pthread_join(*(consumers+i), NULL)) {
			perror("Consumers joining failed\n");
		}
	}

	free(producers);
	producers = NULL;
	free(consumers);
	consumers = NULL;
	pthread_spin_destroy(&spinlock);
}

void* produce (void *arg) {
	while(1) {
		pthread_spin_lock(&spinlock);
		if(producedElements < BUFFER_SIZE) {
			buffer[producerIndex] = rand()%30 + 1;
			producerIndex = (producerIndex+1)%BUFFER_SIZE;
			producedElements++;
		}
		pthread_spin_unlock(&spinlock);
	}
}

void* consume(void *arg) {
	while(1) {
		pthread_spin_lock(&spinlock);
		if(producedElements > 0) {
			printf("Consumed value is %d\n", buffer[consumerIndex]);
			buffer[consumerIndex] = 0;
			consumerIndex = (consumerIndex+1)%BUFFER_SIZE;
			producedElements--;
		}
		pthread_spin_unlock(&spinlock);
	}
}
