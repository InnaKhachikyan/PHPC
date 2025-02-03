#include <stdio.h>

void swap(int* first, int* second);

int main(void) {
	int a = 23;
	int b = 42;
	printf("The value of a is %d.\nThe value of b is %d.\n", a, b);
	swap(&a, &b);
	printf("After the swap: a is %d and b is %d.\n", a, b);

	return 0;
}

void swap(int* first, int* second) {
	int temp = *first;
	*first = *second;
	*second = temp;
}



