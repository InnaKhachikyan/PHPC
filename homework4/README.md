ASSIGNMENT 1:

Array is initialized with some random values.

I have 2 functions naiveLoopMin, which iterates over all the elements and finds the min in O(n) time and simdMin.

simdMin method works as follows:
first loads to some register named minVector the first 8 values of the array.
Then in a loop which starts from 8 (as the first 8 values are already loaded), each time loads 8 values from the array to some register, compares corresponding elements of these registers and stores min of each comparison in the minVector register.
The loop works modulo 8 (meaning if the size is not divisible by 8, we will have some extra elements left).
After exiting the loop, minVector stores 8 ints, which need to be compared within one another.
The left 128 bytes are stored in lowLane 128-register, the right 128 bytes are stored in highlane 128-register.
lowLane and highLane are compared element-wise, and the min result is stored in lowLane 128-register.
These 4 ints are loaded into an array of 4 elements, min is found among these 4 elements.
In the end, we iterate over the remaining elements in the original array (in case SIZE%8 != 0), and compare with the min, to find the absolute minimum from the min (found so far) and the remaining elements.

Running time:
Simd approach runs at least 4 times faster than the iterative one.


ASSIGNMENT 2:

Array is initialized with some random values in range [0,1000].
Naive aproach is iterating over the array and calculating the next value for the output.

SIMD version 1:
it loads weight[0] in weight0 register, weight[1] in weight1 register, and weight[2] in weight2 register.
then each time it loads the array from [i-1] index in the first register, from [i] index in the second, and from [i+1] index in third.

'weight0' register is multiplied with 'first' register, stored in product1.
'weight1' register is multiplied with 'second' register, stored in product2
'weight2' register is multiplied with 'third' register, stored in product3.

Then product1, product2, product3 are added together and the result is stored in the output array

Running time: This simd apporach runs almost twice faster than the naive approach.

SIMD version 2:
I tried to make it more efficient with shift operations.
Each time I load 8 values from the array in a register.
Then using 128 lane shift operations do a shift by 1 int, and store it in another register.
Then again using shift store it in third register.
Like in version 1, calculate the products and sum.
However with this version, each time I am able to calculate 6 outputs only (due to 2 shifts).
I thought it would be faster (as I don't load each time the same data but use shifts), but actually it became slower. 
I think it's due to overhead of the 128 lane operations, and in the version 1 cache maybe helps to bring that same data faster.

Running time: This simd approach runs slower than the naive approach.


ASSIGNMENT 3:

The iterative function is written to work in O(n), not n^2. So it is as fast as it can be (but without application of dynamic programming), and I compare the SIMD version with this O(n) iterative version.

I implemented the SIMD version with shifts: 8 registers store the necessary values to calculate the prefix sum, we load 8 ints, then in the consequent 7 registers we do shifts (in the first one by 1 int, in the second one by 2 ints...), and calculate the sum in a loo.
However, the simd version seems to calculate the sum twice slower than the naive iterative approach.

I implemented version 2 of SIMD (based on my first version AI recommended a faster approach): now I do shift on both left and right 128 lane, calculate the sum, then the sum is shifted by 2 ints (again on both left and right 128 lanes), the result is that the rightmost index of each 128 lane stores the prefix sum of its part. So for the right part we need to also add the prefix sum of the whole left part. that's why I load the rightmost int of the low 128 part to another register and add it to the highest 128 part. In the end the result is added with the output[i-1] (as this index stores the prefix sum calculated so far). This version works almost as fast as the iterative one, 0.4-0.5 seconds slower, but still better than version 1.


ASSIGNMENT 4:

Matrix and vector are initialized with random values in the range [0,10].
The naive approach runs in O(n^2) time.
SIMD:
first we create an array of registers, to store the vector in these registers consequitively, with mod 8 (the remaining elements are processed in the end iteratively).
For each row, we process the column with mod 8 (the rem. is processed separately in the end of the loop).

arg1 stores the next matrix values to be processed, and arg2 next vector values to be processed.
These two registers are multiplied and stored in the product register.
hadd: this is the instruction for horizontal addition, each 2 neighbor values are added together, and the result is stored in the following way: if product[0] = 5 and product[1] = 7, product[2] = 4, product[3] = 6, after hadd product[0]=product[2]=12 and product[1]=product[3]=10;

So, after performing hadd twice, the upper value of the left half will store the whole sum in the leftHalf, and the upper value of the right half will store the whole sum in the rightHalf. It could be done also with the lowest value in each half, no difference.

Running time:
For huge inputs (like the one in the program), simd is nearly 30-35% faster than the naive iterative approach.


ASSIGNMENT 5:




