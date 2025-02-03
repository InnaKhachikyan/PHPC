#include <stdio.h>

int main(void) {
	int arr[5] = {5,5,5,5,5};
	int* pointerToArray = &arr[0];
	int i = 0;
	while(i < sizeof(arr)/4) {
		printf("The value at index %d is %d \n", i, *(pointerToArray + i));
		i++;
	}

	i = 0;
	while(i < sizeof(arr)/4) {
		*(pointerToArray + i) += i + 1;
		i++;
	}

	i = 0;
	printf("*** printing with the pointer ***\n");
	while(i < sizeof(arr)/4) {
		printf("The new value at index %d is %d \n", i, *(pointerToArray + i));
		i++;
	}

	i = 0;
	printf("*** printing with array name ***\n");
	while(i < sizeof(arr)/sizeof(arr[0])) {
		printf("The new value ar index %d is %d \n", i, arr[i]);
		i++;
	} 
	return 0;
}
