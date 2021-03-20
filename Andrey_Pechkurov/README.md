# Comments for the submission

## derrivative_x_y

Simple loop interchange here. Had to extract `r==0` and `r==rows-1` cases into separate loops in order to achieve that.

## gaussian_smooth

Loop interchanges for both X and Y-direction loops. Introduced `float[]` buffers for `dot` and `sum` to be able to do the interchange in the second (Y-direction) loop.

Also split subloops in the first loop into `for(cc=(-center);cc<0;cc++)` and `for(cc=0;cc<=center;cc++)`. This allowed me to decrease iteration ranges in nested loops over c (thanks to the `if(((c+cc) >= 0) && ((c+cc) < cols))` condition).


## apply_hysteresis

Ternary and && -> & are mostly cosmetic (improvements are quite small). The main change here is getting rid of branches in the `hist[mag[pos]]` loop.

## non_max_supp

The list of changes is quite big here:
* Extracted common operations for different branches (e.g. z2 var assignment). Also, introduced z11/z22 vars to provide more opportunities for ILP (they don't depend on z1/z2/mag1 and can be calculated independently)
* Converted the large section with nested branches into a set of lookup tables for z1/z11/z2/z22 pointer offsets + a single `if (gxabs >= gyabs)` branch for mag1/mag2 calculation. I could probably try to convert the whole loop body in a branchless manner, but decided not to do that as the current changes provide a noticeable improvement in terms of branch count decrease
* Simplified last condition in the loop to use a single `if((mag1 <= 0.0) & (mag2 < 0.0))` branch. This also helped to reduce branch count
* Got rid of redundant operations in the loop:
  - Conditional assignment `if(m00 == 0) *resultptr = (unsigned char) NOEDGE` (it's reassigned later in the loop anyway)
  - Division by m00 for xperp and yperp (it's a common divisor for mag1 and mag2, while we're interested in their signs, not exact values)
  - `(gy = *gyptr) >= 0` in the large section with nested branches is not necessary: both gx and gy can be initialized at the beginning of each iteration
