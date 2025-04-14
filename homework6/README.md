#divergenceExperiments.cu

I defined the warp size 32 (as it is standard mainly).

Three global arrays are declared, one for each step.

###initializeArray 
It is a helper function for readability, it just allocates memory, makes a check for malloc and fills the array with value i in a for loop.

###allocCuda, cudaHTDcopy, cudaDTHcopy 
####cudaHTDcopy - memory copy from host to device
####cudaDTH - memory copy from device to host
These are again helper functions for better readability. 
As memory allocations and copies need proper checks with cuda errors (whether the copy was successful, or the allocation), I implemented those snippets in separate functions to call from the host functions instead of repeating that part in each step. 
If allocation or memory copy was not successful the functions take corresponding actions: print error message, free whatever was allocated so far and exit with status 1.

###no_divergence 
This kernel just checks if the index of the thread is within the bounds of the array, multiplies the array element by 2.

###div_kernel 
This kernel checks if the thread index is within the bounds of the array, branches the work: if thread index is divisible by 2, that thread multiplies the current element by 2, otherwise multiplies it by 3.

###min_div 
This kernel again checks whether the thread index is in the bounds of the array, if so alligns the work by the warp size: if index/warpsize is even, do *= 2, else do *=3, this way the first 32 threads will do one branch, the other 32 will do the other branch until all elements are exhausted.

###noDivergence 
This host method calls allocCuda helper method to allocate memory in device. Copies the array with the helper function cudaHTDcopy.
Then it calls the kernel with the predefined threads and blocks number. I chose 256 as it is divisible by 32. And (size + numThreadsPerBlock - 1)/numThreadsPerBlock will ensure that if the elements are not divisible by number of threads one additional block will be created for the remainder.

cudaEvent start and stop are created for fixing the execution time. 
Then I used cudaEventSynchronize instead of cudaDeviceSynchronize to synchronize with the stop time.
synchronization is also done with the error checking.
The execution time is printed.
Array is copied from the device to host memory with the help of cudaDTHcopy function.
And finally corresponding device memory freed.

###divergence, minDivergence
These host methods do the exact steps as noDivergence except for the calls to kernels, each one calls its corresponding kernel.

###main 
The main method initializes the arrays, calls the host functions with corresponding arrays and in the end frees all the allocated memory and sets the pointers to NULL.

##Performance

I increazed the number of elements to observe the difference of performaces better.
Initially, I tried calling no_divergence method first, and saw that it ran slower than expected, the difference between non-divergent and divergent kernel was not so big:

It was more or less like this:
NON-DIVERGENT KERNEL time: 3.519168 ms
DIVERGENT KERNEL time: 3.578304 ms
WARP ALIGNED DIVERGENCE time: 3.342912 ms

After I changed the order: I called the divergent function first, then the other 2 I got this results:
DIVERGENT KERNEL time: 3.723744 ms
NON-DIVERGENT KERNEL time: 3.415360 ms
WARP ALIGNED DIVERGENCE time: 3.344576 ms

Here the difference is more obvious.

However, I called the divergent function twice (one for warmup, the second for the actual results), and the result was this:

DIVERGENT KERNEL time: 3.681120 ms
DIVERGENT KERNEL time: 3.548704 ms
NON-DIVERGENT KERNEL time: 3.307744 ms
WARP ALIGNED DIVERGENCE time: 3.254880 ms

the first call to divergent kernel takes more time than the second call to the same kernel.
And still, divergent kernel is 0.25 ms slower than non-divergent one, and 0.3 ms slower than the warp-aligned one.

