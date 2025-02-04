#include <stdio.h>
#include <stdlib.h>

int main(void) {
	char** pointerToChar = (char**) malloc(sizeof(char*)*5);
	if(!pointerToChar) {
		printf("Memory not allocated!\n");
		return -1;
	}
	*(pointerToChar) = "First";
	*(pointerToChar+1) = "Second";
	*(pointerToChar+2) = "Third";
	*(pointerToChar+3) = "Fourth";
	*(pointerToChar+4) = "Fifth";

	for(int i = 0; i < 5; i++) {
		printf("%s\n", *(pointerToChar+i));
	}

	*(pointerToChar+3) = "MODIFIED";

	for(int i = 0; i < 5; i++) {
		printf("%s\n", *(pointerToChar+i));
	}	
	
	free(pointerToChar);
	return 0;
}
			
