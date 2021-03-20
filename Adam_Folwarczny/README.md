Initial Results:
original:
real    0m2.041s
user    0m1.625s
sys     0m0.359s

Final Results:
Final:
real    0m0.605s
user    0m0.281s
sys     0m0.297s

Compiled with clang++-11 on Windows Subsytem Linux (WSL). CP: i7-8700k.

Final solution compared to original results in 1 difference at location (col:4998; row: 4167). 
My optimized version predicts noedge, reference predicts edge at this location. 


List of optimizations:
1. Memory reuse. (minor)
I have removed allocation of memory for nms and edge variables. 
I reuse memory from variables: smoothedim and image instead.
Very small improvement.
I have also tried to change calloc to malloc but the performance was even worse. No idea why ?

2. Magnitud_xy:
Implementing custom avx2 version. There are 2 versions:
- Resulting exact results with the original code. (commented code)
- Used in the final version, very short nice looking avx code. 
Gives little different results (because of some rounding), but without affecting the final results.

3. derrivative_x_y:
We should process the data in rows then cols to be cache friendly. I have changed it.
Changed the input parameters to __restrict to allow more optimizations.
Removed some poitner arithmetics (*deltax). Finally the code is well vectorized by the compiler automatically.

4. gaussian_smooth
- I have divided the loop into 2 functions: 1 for hotizontal processing (first pass) and another one for vertical processing (second pass)
- From the main loop I have extracted loop which work on the image border to reduce some if statements.
- In the main loops (vertical & horizontal) I used manually written avx2 code to get maximum performance. Compiler wasn't able to generate good enough code.
- In the main loops the "sum" variable is always 1, so we can remove the division completely.
- In the main loops I use fused multiply and add instruction to compute the sums.
- Instead of 2 full passes (horizontal, vertical independently) I use circular buffer to be more cache friendly and to allocate less amount memory. So I process at the beginning few horizontal rows (to fill the circular buffer) and then 1 horizontal pass and 1 vertical pass simultanously reusing memory.

5. apply_hysteresis
- Histogram computation. I have removed the if statement.
		if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++; 
changed to:
		int value = edge[pos] == POSSIBLE_EDGE ? 1 : 0;
        int id = mag[pos];
		hist[id] += value;
- The same trick at the end of this algorithm
- Changed parameter order in if statement (better branch prediction results)

6.  non_max_supp:
Most difficult part:
- (minor) the algorithm is buggy and leaves last 2 columns and last 2 rows with zeros. (it would be enough to do it just for 1 column/row).
- Removed the huge if/else and changed to lookup table. A lot of tiny optimizations to keep the comparisons very simple.
- Optimized the path for m00 == 0.
- removed floating points. Division by m00 is no needed, because we analyze the sign only.


Partial speed ups:

Algorithm		Reference Time[ms]  Final Time[ms]
SmoothGauss:	792,9				73,2
DerivativeXY:	552,2				105,4
MagnitudeXY:	113,1				54,4
NonMaxSup:		254,1				158,0
ApplyHysteresis:211,4				73,9