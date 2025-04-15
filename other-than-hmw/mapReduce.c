#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int wordCount, reducedWordCount;
struct Pair {
	char *key;
	int value;
};

struct Pair *words, *reduced, **hashMap;

void mapWords(char *fileName);
void numberOfWords(char *fileName);
bool isChar(char current);
void reduction();

int main(int argNum, char *args[]) {
	if(argNum < 2) {
		printf("Argument file not given\n");
		return 1;
	}

	char *fileName = args[1];

	mapWords(fileName);
	
	for(int i = 0; i < wordCount; i++) {
		int j = 0;
		while(words[i].key[j] != '\0') {
			printf("%c", words[i].key[j++]);
		}
		printf(" %d\n", words[i].value);
	}
	printf("Number of words: %d\n", wordCount);
	reduction();
	printf("Number of reduced words: %d\n", reducedWordCount);


	for(int i = 0; i < reducedWordCount; i++) {
		int j = 0;
		while(reduced[i].key[j] != '\0') {
			printf("%c", reduced[i].key[j]);
			j++;
		}
		printf(" %d\n", reduced[i].value);
	}
}

void numberOfWords(char *fileName) {
	FILE *file = fopen(fileName, "r");
	if(file == NULL){
		printf("Couldn't open the file\n");
		exit(0);
	}
	wordCount = 0;

	char currentChar;
	bool wordInProgress = false;
	while((currentChar = fgetc(file)) != EOF) {
		if(wordInProgress && !(isChar(currentChar))) {
			wordInProgress = false;
			continue;
		}

		if(!wordInProgress && isChar(currentChar)) {
			wordInProgress = true;
			wordCount++;
		}
	}
	fclose(file);
}

inline bool isChar(char current) {
	if(current < 65 || current > 122 || (current > 90 && current < 97)) {
		return false;
	}
	return true;
}

void mapWords(char *fileName) {

	numberOfWords(fileName);
	words = (struct Pair*)malloc(sizeof(struct Pair)*(wordCount));	
	FILE *file = fopen(fileName, "r");
	if(file == NULL) {
		printf("File not opened\n");
		return;
	}
	char currentChar;
	int i = 0;
	while((currentChar = fgetc(file)) != EOF) {
		if(isChar(currentChar)) {
			words[i].value = 1;
			int j = 0;
			words[i].key = (char*)malloc(128);
			words[i].key[j] = currentChar;
			while((currentChar = fgetc(file)) != EOF) {
				if(isChar(currentChar)) {
					words[i].key[++j] = currentChar;				
				}
				else {
					words[i].key[++j]='\0';
					i++;
					break;
				}
			}
		}
		if(currentChar == EOF) {
			break;
		}
	}
}

void reduction() {
    reduced = (struct Pair*)malloc(sizeof(struct Pair)*wordCount);
    int reducedCount = 0;  
    for(int i = 0; i < wordCount; i++) {
        if(words[i].value != -1) {  
            reduced[reducedCount].key = (char*)malloc(strlen(words[i].key) + 1);
	    int k;
	    for(k = 0; k < strlen(words[i].key); k++) {
		    reduced[reducedCount].key[k] = words[i].key[k];
		}
	    reduced[reducedCount].key[k] = '\0';
            //strcpy(reduced[reducedCount].key, words[i].key);
            reduced[reducedCount].value = words[i].value;  
            for(int j = i + 1; j < wordCount; j++) {
                if(words[j].value != -1) {  
                    if(strcmp(words[i].key, words[j].key) == 0) {
                        reduced[reducedCount].value += words[j].value;
                        words[j].value = -1;  
                    }
                }
            }
            reducedCount++;  
        }
    }
    reduced = (struct Pair*)realloc(reduced, sizeof(struct Pair) * reducedCount);
    reducedWordCount = reducedCount;
}











