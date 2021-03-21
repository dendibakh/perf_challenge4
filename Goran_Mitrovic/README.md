Developed with Visual Studio 2019 on i7-7700K using VCL 2.01.03 by Agner Fog (should be downloaded separately).

'canny' routine timings measured at my place:
original: 2550ms
MSVC: 450ms
clang10 in Ubuntu VM: 680ms


List of optimizations (I have unrolled loops heavily by hand, so the source code is often not very readable):
1. gaussian_smooth
 - both loops vectorized by hand
 - second loop is done in columns, but only half cache line large
 - eliminated divisions
 - calculated derrivative_y for free
 
2. derrivative_x_y
 - vectorized by hand
 - skipped y as it was calculated in the previous routine
 
3. magnitude_x_y
 - vectorized by hand
 
4. non_max_supp
 - eliminated all conditional jumps; only 4 CMOVs with MSVC compiler (haven't checked what Clang generates)
 - eliminated division
 - no lookup tables

5. apply_hysteresis
 - some basic optimizations; merging some loops together

6. follow_edges
 - eliminated recursion using fifo buffer (variable name suggests it's circular buffer, but I got rid of it at some point)
 - eliminated all conditional jumps (no speedup there, but looked nice)
