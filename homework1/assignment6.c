#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int str_length(char* str);

int main(void) {
	char* pointerToString = "Hello world";
	for(int i = 0; (*(pointerToString+i)) != '\0'; i++) {
		printf("%c",*(pointerToString+i));
	}
	printf("\n");

	char* pointerToInput = (char*) malloc(sizeof(char)*10);
	if(!pointerToInput) {
		printf("Memory allocation failed!\n");
		return -1;
	}
	printf("Enter a text!\n");
	scanf("%s", pointerToInput);

	printf("The length of the input string is %d\n", str_length(pointerToInput));
	return 0;
}

int str_length(char* str) {
	if(str) {
		int length = 0;
		while(*(str+length) != '\0') {
			length++;
		}
		return length;
	}
	return -1;
}
