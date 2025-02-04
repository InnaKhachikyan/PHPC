#include <stdio.h>
#include <stdlib.h>

int main(void) {
	int* pointerToInt = (int*) malloc(sizeof(int));
	
	if(!pointerToInt) {
		printf("Memory for int not allocated!\n");
		return -1;
	}

	*pointerToInt = 23;
	printf("%d\n", *pointerToInt);
	
	int* pointerToArray = (int*) malloc(sizeof(int)*5);
	
	if(!pointerToArray) {
		printf("Memory for array not allocated!\n");
		return -1;
	}
	
	int arrayLength = 5;
	for(int i = 0; i < arrayLength; i++) {
		*(pointerToArray+i) = i;
		printf("index %d is %d\n",i, *(pointerToArray+i));
	}

	free(pointerToInt);
	free(pointerToArray);

	return 0;
}
