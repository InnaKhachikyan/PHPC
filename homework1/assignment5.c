#include <stdio.h>
#include <stdlib.h>

int main(void) {
	int* ptr = (int*) malloc(sizeof(int));
	*ptr = 23;
	printf("%d\n", *ptr);
	
	int* arrPtr = (int*) malloc(sizeof(int)*5);
	int length = 5;
	for(int i = 0; i < length; i++) {
		*(arrPtr+i) = i;
		printf("index %d is %d\n",i, *(arrPtr+i));
	}

	free(ptr);
	free(arrPtr);

	return 0;
}
