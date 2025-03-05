link:

https://github.com/InnaKhachikyan/PHPC/tree/main/homework3

Assignment 1:
The struct player has playerNumber, rolledDice (which stores the dice that was rolled for that player most recently) and score which is incremented if the player has won the current round. The game is summed up according to the score (the player who has the highest score wins). If 2 or more players have equal highest rolledDice, nobody wins. One barrier is put after rolling the dice, so that everyone waits untill all the threads have rolled the dice. The second barrier is put after if condition (if playerNumber == 1), this ensures that all the other threads wait at the barrier while the first thread sums up the winner of the round (so that nobody rolls dice until the round winner is summed up). Then the first thread reaches the barrier after summing up the roundWinner after the if condition and the next round begins.

Assignment2:
Every player struct has a number and a sleepTime, which is generated randomly by rand() function (in the range [1,10]. The barrier is put one in the get_Ready function so that all the threads gather there, and one in the play. The barrier is initialized for  NUMBER_OF_PLAYERS+1 threads (+1 is for the main thread, it also waits at the barrier till everyone is ready then calls startGame()).

Assignment 3:
I created NUMBER_OF_SENSORS + 1 threads, the last thread is meant for calculating the average of all the data. All the other threads work with the readTemperature function (where a random number in range [-25,45] is generated), the last thread works with the countAverage function where it calculates the average of the collected data.
I have put 2 barriers:
First one is for waiting that everyone collects their data before the last thread calculates the average. The second barrier is for the threads to wait until the average is calculated before reading new data and updating it in the structs.

I also added sleep(3) in the readTemperature so that the process (printing on the screen) goes a little bit slow for visibility.

Assignment 4:
I implemented a program which calculates the expression (vector1 + vector2 - vector3) dotProduct vector4 in 3 stages: addition, subtraction and dot product.
After each stage the threads wait for each other to proceed to the next calculation.
There are two approaches:
1. perform all the stages calculation in the function pipeline
2. Call in one function the next step each time and finalize the result in the function calculateExpression where the threads were created.
One version is commented, but it works too.
I added srand() which gives as seed the time in seconds to random generator, without that I wlways got the same values for the arrays (I don't know yet how the random generator in C works, didn't figure out yet, so by your advice I added srand()).

 
