#include <stdio.h>

int main(void) {
	int var;
	int* pointerToVar = &var;
	int** pointerToPointerToVar = &pointerToVar;
	**pointerToPointerToVar = 23;
	printf("Printing 'var' with the pointer: %d\n", *pointerToVar);
	printf("Printing 'var' with the double pointer: %d\n", **pointerToPointerToVar);

	return 0;
}
