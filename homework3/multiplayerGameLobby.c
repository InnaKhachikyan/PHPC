#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#define NUMBER_OF_PLAYERS 8

struct player* players;
pthread_barrier_t barrier;

struct player {
	int playerNumber;
	int sleepTime;
};

void startGame(void);
void play(struct player* players);
void* get_ready(void* arg);

int main(void) {
	players = (struct player*)malloc(sizeof(struct player)*NUMBER_OF_PLAYERS);
	if(players == NULL) {
		printf("Memory allocation for the players failed!\n");
		return 1;
	}

	for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
		players[i].playerNumber = i + 1;
		players[i].sleepTime = rand()%10 + 1;
	}

	play(players);

	free(players);
}

void play(struct player* players) {
	pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*NUMBER_OF_PLAYERS);
	if(threads == NULL) {
		printf("Memory allocation for threads failed!\n");
		exit(1);
	}
	pthread_barrier_init(&barrier,NULL,(NUMBER_OF_PLAYERS+1));	
	for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
		if(pthread_create((threads+i),NULL,&get_ready,(players + i))){
			perror("Thread creating failed!\n");
		}
	}
	printf("*** THREAD NUMBER %lu WAITING ***\n",(unsigned long)pthread_self());
	
	pthread_barrier_wait(&barrier);

	startGame();

	for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
		if(pthread_join(*(threads+i),NULL)) {
			perror("Thread join failed!\n");
		}
	}

	free(threads);
}

void* get_ready(void* arg) {
	struct player* currentPlayer = (struct player*)arg;
	sleep(currentPlayer->sleepTime);
	printf("*** THREAD NUMBER %lu GETTING READY ***\n",(unsigned long)pthread_self());
	pthread_barrier_wait(&barrier);
}

void startGame() {
	printf("^^^^ STARTING THE GAME ^^^^\n");
}
