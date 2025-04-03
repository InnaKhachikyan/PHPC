long fibonacciDP(int n)

It is a dynamic programming method. I created that just for checking my omp method calculations are correct.
I don't use the standard recursive method (fibonacciRec) for checking as it is very slow, I just wrote it to visualize what should be made parallel with OMP.



long fibonacciOmp(int n)

In the method there is a region parallel which creats multiple threads to execute the parallel region. Inside the parallel region one thread calls the fibonacciTask recursive method with the argument n. single omp ensures that only one thread makes that call.
When that single thread calls that method, it may either go to calculate F(num) iteratively in the base case if n<=10 or if n>10 it will go further to create tasks.
At first the task is created by the single thread that called the recursive function.
Let's say n = 20;
It reaches the task shared(fib1) part, creates a task for n = 19 and n = 18, adds to the tasks queue. Some thread, let's say thread number 5, takes the task (F19) from the queue makes the recursive call fibonaccitask(19) enters the method, skips the base case and goes further creates the tasks for n = 18 and n = 17. 
The task creation is done recursively, so the threads that took the task from the queue recursively create tasks for smalled n and add those to queue.
So, the thread creating the task is not always the same (it's the thread that took the next task from the pool and reached the line of createing a new task). 
The tasks are added to a queue and the next task in the queue is executed by any available thread at the moment.



shared(fib1) and shared(fib2)

this part ensures that fib1 and fib2 are shared variables, so the parent thread that created the tasks for n-1 and n-2 will have access to the final calculations made by other threads that took the tasks and calculated fib1 and fib2.



taskwait

this part ensures threads synchronization, we wait for the two threads executing n-1 and n-2 to finish their calculations and give the updated fib1 and fib2 so that the parent thread may calculate their sum.


usleep and printings are added just for better visualization of the execution.
