#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

long result;
long fibonacciDP(int n);
long fibonacciOmp(int n);
long fibonacciTask(int n);

int main() {

	long resultDP, resultOmp;
	resultDP = fibonacciDP(10);
	printf("FINAL RESULT DP n=10: %ld\n", resultDP);
	resultOmp = fibonacciOmp(10);
/*	#pragma omp parallel 
	{ 
		#pragma omp single
		{
			resultOmp = fibonacciOmp(10);
		}
	}
*/	printf("FINAL RESULT OMP n=10: %ld\n", resultOmp);

	resultDP = fibonacciDP(15);
	printf("FINAL RESULT DP n=15: %ld\n", resultDP);
	resultOmp = fibonacciOmp(15);
/*	#pragma omp parallel 
	{ 
		#pragma omp single
		{
			resultOmp = fibonacciOmp(15);
		}
	}
*/	printf("FINAL RESULT OMP n=15: %ld\n", resultOmp);


	return 0;
}

//to check that the OMP result is the correct
long fibonacciDP(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    long dp[n + 1];  // Array to store Fibonacci numbers
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];  // Compute Fibonacci iteratively
    }

    return dp[n];
}

long fibonacciOmp(int n) {
	long finalResult;
	#pragma omp parallel
	{
		#pragma omp single
		{
			finalResult = fibonacciTask(n);
		}
	}
	return finalResult;
}

long fibonacciTask(int n) {
	long fib1, fib2, result;
	if(n <= 10) {
		fib1 = 0;
		fib2 = 1;
		for(int i = 2; i <= n; i++) {
			result = fib1 + fib2;
			fib1 = fib2;
			fib2 = result;
		}
		printf("IN BASE CASE result %ld, n is %d\n", result, n);
		return result;
	}

	else {	
		#pragma omp task shared(fib1)
		{
			printf("N is %d\n", n);
			fib1 = fibonacciOmp(n-1);
			printf("FIB1 is %ld, n is %d\n", fib1, n);
			printf("number of thread %d\n", omp_get_thread_num());
			//usleep(100000);
		}
		#pragma omp task shared(fib2)
		{
			printf("N is %d\n", n);
			fib2 = fibonacciOmp(n-2);
			printf("FIB2 is %ld, n is %d\n", fib2, n);
			//usleep(100000);
		}
		#pragma omp taskwait
		printf("RESULT %ld \n", (fib1 + fib2));
		return fib1 + fib2;
		//usleep(1000000);
	}
}

