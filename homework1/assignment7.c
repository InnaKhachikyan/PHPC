#include <stdio.h>

int main(void) {
	char* arrayOfStrings[5];
	arrayOfStrings[0] = "First";
	arrayOfStrings[1] = "Second";
	arrayOfStrings[2] = "Third";
	arrayOfStrings[3] = "Fourth";
	arrayOfStrings[4] = "Fifth";

	for(int i = 0; i < 5; i++) {
		printf("%s\n", arrayOfStrings[i]);
	}

	arrayOfStrings[3] = "MODIFIED";

	for(int i = 0; i < 5; i++) {
		printf("%s\n", arrayOfStrings[i]);
	}	

	return 0;
}
			
