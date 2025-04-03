#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

long fibonacciRec(int n);
long fibonacciDP(int n);
long fibonacciOmp(int n);
long fibonacciTask(int n);

int main() {

	long resultDP, resultOmp;
	int n = 10;
	resultDP = fibonacciDP(n);
	printf("FINAL RESULT DP n = %d: %ld\n", n, resultDP);
	resultOmp = fibonacciOmp(n);
	printf("FINAL RESULT OMP n = %d: %ld\n", n, resultOmp);

	n = 15;
	resultDP = fibonacciDP(n);
	printf("FINAL RESULT DP n = %d: %ld\n", n, resultDP);
	resultOmp = fibonacciOmp(n);
	printf("FINAL RESULT OMP n = %d: %ld\n", n, resultOmp);


	return 0;
}

//to check that the OMP result is the correct
long fibonacciDP(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    long dp[n + 1];  
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];  
    }

    return dp[n];
}

long fibonacciRec(int n) {
	if(n == 0) return 0;
	if(n == 1) return 1;

	long result = fibonacciRec(n-1) + fibonacciRec(n-2);
	return result;
}
	

long fibonacciOmp(int n) {
	long finalResult;
	#pragma omp parallel
	{
		#pragma omp single
		{
			printf("Thread number %d calls recursive Fibonacci\n", omp_get_thread_num());
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
		return result;
	}

	else {	
		usleep(100000);
		printf("Thread number %d creates task %d\n", omp_get_thread_num(), n-1);
		#pragma omp task shared(fib1)
		{
			printf("Thread number %d processes %d-th Fibonacci\n", omp_get_thread_num(), n);
			fib1 = fibonacciTask(n-1);
			usleep(100000);
		}
		usleep(100000);
		printf("Thread number %d creates task %d\n", omp_get_thread_num(), n-2);
		#pragma omp task shared(fib2)
		{
			printf("Thread number %d processes %d-th Fibonacci\n", omp_get_thread_num(), n);
			fib2 = fibonacciTask(n-2);
			usleep(100000);
		}
		#pragma omp taskwait
		printf("RESULT %ld \n", (fib1 + fib2));
		return fib1 + fib2;
	}
}

