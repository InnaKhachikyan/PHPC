


Assignment 2:

When loading the array in 3 registers (one starting from index i-1, the second from index i, and the third from index i+1) SIMD is not as effective as expected. It runs nearly twice faster than the iterative version.

Assignment 3:

I implemented the SIMD version with shifts: 8 registers store the necessary values to calculate the prefix sum, we load 8 ints, then in the consequent 7 registers we do shifts (in the first one by 1 int, in the second one by 2 ints...), and calculate the sum in a loo.
However, the simd version seems to calculate the sum twice slower than the scalar.

I implemented version 2 of SIMD (based on my first version AI recommended a faster approach): now I do shift on both left and right 128 lane, calculate the sum, then the sum is shifted by 2 ints (again on both left and right 128 lanes), the result is that the rightmost index of each 128 lane stores the prefix sum of its part. So for the right part we need to also add the prefix sum of the whole left part. that's why I load the rightmost int of the low 128 part to another register and add it to the highest 128 part.In the end the result is added with the output[i-1] (as this index stores the prefix sum calculated so far. This version works almost as fast as the iterative one, 0.4-0.5 seconds slower, but still better than version 1.


