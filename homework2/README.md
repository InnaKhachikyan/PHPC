I added scripts for running the programs in a loop 100 times, each script is meant for one program).
I also added a makefile, which compiles all the source files and creates corresponding executables to run with the scripts.

globalVarThreadedSine.c
This program has a race condition: all the threads work with the global variableand add their calculations there. I created a function which creates and executes multiple threads (the number of threads is passed as an argument). The runLoop.sh runs the executable globalVarThreadedSine 100 times and collects the results in the output.txt. However there is no damaged result because of the race condition (neither with 4 threads, nor with 8), and I don't know yet why. 

thrededSine.c
In this program sine function is calculated with 4 and 8 threads (again with the help of a function which creats N number of threads), but here the threads do their part of the job, then join their calculations together.
The runLoop2.sh runs the executable threadedSine 100 times and collects the results in the output2.txt. By the results there we see that 8 threads calculate much faster and in both cases the results are correct.

sineWithSpinlock and sineWithMutex
To tell the truth I expected that the spinlock version would be faster (as the global variable is locked for a very short period and I thought Mutex would have an overhead of context switch). However as the output files show (those store the results of 100 executions in a loop), the results don't vary that much.
