#include <stdio.h>
#include <time.h>
#define PI 3.14159265358979323846

double degreesToRadians(double degree);
double factorial(int n);
double power(double x, int n);
double sine(double x, int n);

double degreesToRadians(double degree) {
	return degree * (PI/180.0);
}

double factorial(int n) {
	double factorial = 1;
	for(int i = 1; i <= n; i++) {
		factorial = factorial * i;
	}
	return factorial;
}

double power(double x, int n) {
	if(n == 0) {
		return 1;
	}
	double partial = power(x, (int)n/2);
	double result = partial * partial;
	if(n%2 == 1) {
		result *= x;
	}
	return result;
}

double sine(double x, int n) {
	double result = 0.0;
	x = degreesToRadians(x);

	for(int i = 0; i <= n; i++) {

		result += (((i%2 == 0) ? 1 : (-1))/(factorial(2*i+1)))*power(x,(2*i+1));
	}
	return result;
}

int main(void) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	printf("Sine of 0 is: %f\n", sine(0, 40));
	printf("Sine of 30 is: %f\n", sine(30,20));
	printf("Sine of 45 is: %f\n", sine(45,40));
	printf("Sine of 60 is: %f\n", sine(60,40));
	printf("Sine of 90 is: %f\n", sine(90,40));
	clock_gettime(CLOCK_MONOTONIC, &end);
	double total = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9);
	printf("Execution time: %f\n",total);
	
	return 0;
}
