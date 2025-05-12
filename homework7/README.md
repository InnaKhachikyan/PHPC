### dna.cu

I created helper methods to make the code more readable:

initializeArray method initializes the data array with dna values randomly.

cleanup function: checks if each dynamically allocated memory is not null, frees it, and sets pointer to null.

memoryAllocAndCopy function: allocates memory on host and devices sides, as well as copies from the host to device and makes all the necessary checks, calls the cleanup function if the program has to be stopped.

countDna kernel: accepts data as argument, creates local variables, so each thread will calculate its local sum for each letter. The loop is grid-stride. In the end the local sum of each letter is added to the corresponding cell of the result array with atomicAdd function, which ensures no race conditions occur.

In the main method countDna is called twice (the first one is for warmup).
Time measurements show that the kernel works 12-13 times faster than the cpu on 1 million elements.

Execution example:
GPU TIME: 0.204000
CPU TIME: 2.491000
GPU 12.210784 times faster

### rna.cu

the rna program works very similar to dna, the kernel is with a grid-stride loop, the same helper functions, kernel called twice in the main for warmup.
However the kernel has a simpler job: it just passes through the array, and wherever any thread encounters letter 'T' it is changed to 'U'. First I was copying to some output array, but it was slow, so I changed it so that the kernel does the change directly in the argument array if necessary.
The same logic is written for the CPU, for sake of 'fair' time measurement, so the CPU also changes the input data if necessary.
The time measurements show that the kernel works 135 times faster.

Execution example:
GPU TIME: 0.043000
CPU TIME: 5.820000
GPU is 135.348837 times faster

### pointMutation.cu

I have the same helper methods here. 
In this kernel each thread handles up to 8 elements: each thread loads up to 8 data with corresponding indices (each block-stride apart) and accumulates 1 in the local var partial with boolean true if it finds a mismatch. The shared array partialSum is declared per block (with dynamic size of numbers of threads per block), so each thread writes its accumulated partial in its corresponding index with tid, then the elements of the partialSum are added together, and the total sum is calculated with reduction and final warp unrolling.

First I hard coded the number of threads as the size of partialSum array (partialSum[256]), and the time measurements showed that the kernel was 120-124 times faster:

Execution example:
GPU TIME: 0.025000
CPU TIME: 3.026000
GPU is 121.040000 times faster

However, whan I changed it to a dynamic size, which is a better implementation if the number of threads need to be changed, the time measurements showed that the kernel now is 100 times faster:
Execution example:
GPU TIME: 0.026000
CPU TIME: 2.631000
GPU is 101.192308 times faster

Without the template it was only 90 times faster.


### proteinMass.cu

I have the same helper methods here. Additionally, two methods to create the alphabet and protein.
As in the popintMutation program each thread handles up to 8 elements. I used reduction and unrolling with template and final warps unrolling.

First I hardcoded the shared array partialSum with the number of threads, but then changed it to dynamic size. 
The kernel works up to 53 times faster.
Execution example:
GPU TIME: 0.595000
CPU TIME: 32.073000
GPU is 53.904202 times faster

### rnaToProtein.cu

I tried to use a fast method to find the corresponding char for the three letters: I sorted the three lettered values in alphabetical order, created the table with corresponding values in the constant memory. As we have only 4 letters, I used those as index shortcuts: A stands for 0, C stands for 1, G stands for 2 and U stands for 3. Not to overcomplicate the kernel code for this encodings, I used forceline (which works the same way as plain C inline). Each letter in the triplet is a part of index in the table. The first letter is read, replaced with corresponding number and shifted by 4 positions, then the second letter is read, replaced with corresponding number and shifted by 2 positions, and the last letter is read and replaced with its values. These 3 letters numbered values are OR-ed together, so their sum shows us at which index in the table is the corresponding value.
In the kernel I copy the table from constant memory to shared, so that each block has its own copy and takes the corresponding values quickly from on-chip memory.
The kernel works up to 58 times faster.

Execution example:
GPU time: 0.337000
CPU time: 19.591000
GPU is 58.133531 times faster
