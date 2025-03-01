#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUMBER_OF_PLAYERS 8
#define GAME_ROUNDS 6

pthread_barrier_t barrier;
struct player* players;
struct player* play(struct player* players);
void* playerRoll(void* arg);
void sumUpRoundWinner(struct player* players);
struct player*  sumUpGameWinner(struct player* players);

struct player {
    int playerNumber;
    int rolledDice;
    int score;
};

int main(void) {
    players = (struct player*)malloc(sizeof(struct player)*NUMBER_OF_PLAYERS);
    if(players == NULL) {
        printf("Memory allocation for players failed!\n");
	    return 1;
    }

    for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
	    players[i].playerNumber = i + 1;
	    players[i].rolledDice = 0;
	    players[i].score = 0;
    }

    struct player *winner = play(players);
    if(winner != NULL) {
        printf("The winner is number %d player\n", winner->playerNumber);
    }
    else {
	    printf("Nobody wn the game!\n");
    }
    free(players);
}

// play NUMBER_OF_ROUNDS times and return the winner
struct player* play(struct player* players) {
    pthread_t* playerThreads = (pthread_t*)malloc(sizeof(pthread_t)*NUMBER_OF_PLAYERS);
    if(playerThreads == NULL) {
        printf("Memory allocation for threads failed!\n");
    }
    pthread_barrier_init(&barrier, NULL, NUMBER_OF_PLAYERS);
    for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
        printf("*** CRETING THE THREAD NUMBER %d ***\n", i+1);
        if(pthread_create((playerThreads+i), NULL,&playerRoll, (players +i))) {
            perror("Thread creating failed!\n");
        }
    }

    for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
        if(pthread_join(*(playerThreads+i),NULL)) {
            perror("Thread join failed!\n");
        }
    }
    free(playerThreads);
    pthread_barrier_destroy(&barrier);
    return sumUpGameWinner(players);
}

void sumUpRoundWinner(struct player* players) {
    int winnerIndex = -1;
    int maxRoll = 0;
    int numberOfWinners = 0;
    for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
        struct player currentPlayer = players[i];
        if(currentPlayer.rolledDice > maxRoll){
            maxRoll = currentPlayer.rolledDice;
            numberOfWinners = 1;
            winnerIndex = i;
        }
        else if(currentPlayer.rolledDice == maxRoll) {
            numberOfWinners++;
        }
    }
    if(numberOfWinners == 1) {
        players[winnerIndex].score += 1;
        printf("Player number %d won current round!\n", players[winnerIndex].playerNumber);
    }
    else {
	    printf("Nobody won current round!\n");
    }
}

struct player* sumUpGameWinner(struct player* players) {
    int winnerIndex = -1;
    int highestScore = 0;
    int numberOfWinners = 0;

    for(int i = 0; i < NUMBER_OF_PLAYERS; i++) {
        if(players[i].score > highestScore) {
            winnerIndex = i;
            numberOfWinners = 1;
            highestScore = players[i].score;
        }
        else if(players[i].score == highestScore) {
            numberOfWinners += 1;
        }	    
    }

    if(numberOfWinners > 1) {
        return NULL;
    }
    return &players[winnerIndex];
}

void* playerRoll(void* arg) {
    struct player* currentPlayer = (struct player*) arg;
    for(int i = 0; i < GAME_ROUNDS; i++) {
        printf("   ***   ROUND NUMBER %d   ***\n", i+1);
        currentPlayer->rolledDice = rand()%6 + 1;
        printf("!!! WAITING AT THE BARRIER Thread id: %lu !!!\n", (unsigned long)pthread_self());
        pthread_barrier_wait(&barrier);
        // This if condition is to ensure that only one thread sums up the round
        if(currentPlayer->playerNumber == 1) {
            sumUpRoundWinner(players);
        }
	    pthread_barrier_wait(&barrier);
    }
    return NULL;
}
