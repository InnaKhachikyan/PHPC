#include <stdio.h>

int main(void) {
	int var = 23;
	int* pointerToVar = &var;
	printf("The address of var using & is %p\n", &var);
	printf("The address of var using pointer is %p\n", pointerToVar);

	//modifying the value using the pointer
	(*pointerToVar)=42;
	printf("The new value of var is: %d\n", var);
	return 0;
}
