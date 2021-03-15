/*******************************************************************************
* --------------------------------------------
*(c) 2001 University of South Florida, Tampa
* Use, or copying without permission prohibited.
* PERMISSION TO USE
* In transmitting this software, permission to use for research and
* educational purposes is hereby granted.  This software may be copied for
* archival and backup purposes only.  This software may not be transmitted
* to a third party without prior permission of the copyright holder. This
* permission may be granted only by Mike Heath or Prof. Sudeep Sarkar of
* University of South Florida (sarkar@csee.usf.edu). Acknowledgment as
* appropriate is respectfully requested.
* 
*  Heath, M., Sarkar, S., Sanocki, T., and Bowyer, K. Comparison of edge
*    detectors: a methodology and initial study, Computer Vision and Image
*    Understanding 69 (1), 38-54, January 1998.
*  Heath, M., Sarkar, S., Sanocki, T. and Bowyer, K.W. A Robust Visual
*    Method for Assessing the Relative Performance of Edge Detection
*    Algorithms, IEEE Transactions on Pattern Analysis and Machine
*    Intelligence 19 (12),  1338-1359, December 1997.
*  ------------------------------------------------------
*
* PROGRAM: canny_edge
* PURPOSE: This program implements a "Canny" edge detector. The processing
* steps are as follows:
*
*   1) Convolve the image with a separable gaussian filter.
*   2) Take the dx and dy the first derivatives using [-1,0,1] and [1,0,-1]'.
*   3) Compute the magnitude: sqrt(dx*dx+dy*dy).
*   4) Perform non-maximal suppression.
*   5) Perform hysteresis.
*
* The user must input three parameters. These are as follows:
*
*   sigma = The standard deviation of the gaussian smoothing filter.
*   tlow  = Specifies the low value to use in hysteresis. This is a 
*           fraction (0-1) of the computed high threshold edge strength value.
*   thigh = Specifies the high value to use in hysteresis. This fraction (0-1)
*           specifies the percentage point in a histogram of the gradient of
*           the magnitude. Magnitude values of zero are not counted in the
*           histogram.
*
* NAME: Mike Heath
*       Computer Vision Laboratory
*       University of South Floeida
*       heath@csee.usf.edu
*
* DATE: 2/15/96
*
* Modified: 5/17/96 - To write out a floating point RAW headerless file of
*                     the edge gradient "up the edge" where the angle is
*                     defined in radians counterclockwise from the x direction.
*                     (Mike Heath)
*******************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <intrin.h>

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth_derivative_magnitude(unsigned char *image, int rows, int cols, float sigma,
    short int* delta_x, short int* delta_y, short int* magnitude);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void apply_hysteresis(short int *mag, int rows, int cols,
        float tlow, float thigh, unsigned char *edge);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
    unsigned char *result);

int main(int argc, char *argv[])
{
   char *infilename = NULL;  /* Name of the input image */
   char *dirfilename = NULL; /* Name of the output gradient direction image */
   char outfilename[128];    /* Name of the output "edge" image */
   char composedfname[128];  /* Name of the output "direction" image */
   unsigned char *image;     /* The input image */
   unsigned char *edge;      /* The output edge image */
   int rows, cols;           /* The dimensions of the image. */
   float sigma,              /* Standard deviation of the gaussian kernel. */
	 tlow,               /* Fraction of the high threshold in hysteresis. */
	 thigh;              /* High hysteresis threshold control. The actual
			        threshold is the (100 * thigh) percentage point
			        in the histogram of the magnitude of the
			        gradient image that passes non-maximal
			        suppression. */

   /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
   if(argc < 5){
   fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
      fprintf(stderr,"\n      image:      An image to process. Must be in ");
      fprintf(stderr,"PGM format.\n");
      fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
      fprintf(stderr," blur kernel.\n");
      fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
      fprintf(stderr,"edge strength threshold.\n");
      fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
      fprintf(stderr," of non-zero edge\n                  strengths for ");
      fprintf(stderr,"hysteresis. The fraction is used to compute\n");
      fprintf(stderr,"                  the high edge strength threshold.\n");
      fprintf(stderr,"      writedirim: Optional argument to output ");
      fprintf(stderr,"a floating point");
      fprintf(stderr," direction image.\n\n");
      exit(1);
   }

   infilename = argv[1];
   sigma = atof(argv[2]);
   tlow = atof(argv[3]);
   thigh = atof(argv[4]);

   if(argc == 6) dirfilename = infilename;
   else dirfilename = NULL;

   /****************************************************************************
   * Read in the image. This read function allocates memory for the image.
   ****************************************************************************/
   if(VERBOSE) printf("Reading the image %s.\n", infilename);
   if(read_pgm_image(infilename, &image, &rows, &cols) == 0){
      fprintf(stderr, "Error reading the input image, %s.\n", infilename);
      exit(1);
   }

   /****************************************************************************
   * Perform the edge detection. All of the work takes place here.
   ****************************************************************************/
   if(VERBOSE) printf("Starting Canny edge detection.\n");
   if(dirfilename != NULL){
      sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
      sigma, tlow, thigh);
      dirfilename = composedfname;
   }
   canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

   /****************************************************************************
   * Write out the edge image to a file.
   ****************************************************************************/
   sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename,
      sigma, tlow, thigh);
   if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
   if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0){
      fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
      exit(1);
   }

   return 0;
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{
   short int *delta_x,        /* The first devivative image, x-direction. */
             *delta_y,        /* The first derivative image, y-direction. */
             *magnitude;      /* The magnitude of the gadient image.      */

   if ((delta_x = (short*)malloc(sizeof(short) * rows * cols)) == NULL) {
       fprintf(stderr, "Error allocating the delta_x image.\n");
       exit(1);
   }
   if ((delta_y = (short*)malloc(sizeof(short) * rows * cols)) == NULL) {
       fprintf(stderr, "Error allocating the delta_y image.\n");
       exit(1);
   }
   if ((magnitude = (short*)malloc(sizeof(short) * rows * cols)) == NULL) {
       fprintf(stderr, "Error allocating the magnitude image.\n");
       exit(1);
   }
   if ((*edge = (unsigned char*)malloc(sizeof(unsigned char) * rows * cols)) == NULL) {
       fprintf(stderr, "Error allocating the edge image.\n");
       exit(1);
   }

   /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
   if (VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");

   gaussian_smooth_derivative_magnitude(image, rows, cols, sigma,
       delta_x, delta_y, magnitude);

   /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/
   if(VERBOSE) printf("Doing the non-maximal suppression.\n");

   non_max_supp(magnitude, delta_x, delta_y, rows, cols, *edge);

   /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
   if(VERBOSE) printf("Doing hysteresis thresholding.\n");

   apply_hysteresis(magnitude, rows, cols, tlow, thigh, *edge);

   /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
   free(magnitude);
   free(delta_y);
   free(delta_x);
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth_derivative_magnitude(unsigned char *image, int rows, int cols, float sigma,
        short int* delta_x, short int* delta_y, short int* magnitude)
{
   int r, c, rr, cc, pos,/* Counter variables. */
      windowsize,        /* Dimension of the gaussian kernel. */
      center;            /* Half of the windowsize. */
   float *kernel,        /* A one dimensional gaussian kernel. */
         dot, *doty, *dotx, /* Dot product summing variable. */
         sum, k;         /* Sum of the kernel weights variable. */
   double scale;
   short int *z2, *z1, *z0, *swap;
   int dx, dy;

   /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);
   center = windowsize / 2;

   /****************************************************************************
   * Allocate a temporary buffer image and the smoothed image.
   ****************************************************************************/
   if((doty = (float *) malloc(sizeof(float)*cols)) == NULL){
      fprintf(stderr, "Error allocating the row buffer.\n");
      exit(1);
   }
   if((dotx = (float *) malloc(sizeof(float)*cols)) == NULL){
      fprintf(stderr, "Error allocating the row buffer.\n");
      exit(1);
   }
   if((z2 = (short *) malloc(sizeof(short)*cols)) == NULL){
      fprintf(stderr, "Error allocating the row buffer.\n");
      exit(1);
   }
   if((z1 = (short *) malloc(sizeof(short)*cols)) == NULL){
      fprintf(stderr, "Error allocating the row buffer.\n");
      exit(1);
   }
   if((z0 = (short *) malloc(sizeof(short)*cols)) == NULL){
      fprintf(stderr, "Error allocating the row buffer.\n");
      exit(1);
   }

   if(VERBOSE) printf("   Bluring the image.\n");
      for(r=0;r<rows;r++){
         /****************************************************************************
         * Blur in the y - direction.
         ****************************************************************************/
         sum = 0.0f;
         rr = max(-r, -center);
         {
               k = kernel[center + rr];
               pos = (r+rr)*cols;
               for(c=0;c<cols;c++){
                  doty[c] = (float)image[pos+c] * k;
               }
               sum += k;
         }
         while (++rr <= min(center,rows-1-r)) {
               k = kernel[center + rr];
               pos = (r+rr)*cols;
               for(c=0;c<cols;c++){
                  doty[c] += (float)image[pos+c] * k;
               }
               sum += k;
         }

         scale = BOOSTBLURFACTOR/sum;

      /****************************************************************************
      * Blur in the x - direction.
      ****************************************************************************/
      for(c=0;c<center;c++){
         dot = 0.0f;
         sum = 0.0f;
         for(cc=-c;cc<=center;cc++){
               dot += doty[c+cc] * kernel[center+cc];
               sum += kernel[center+cc];
         }
         z0[c] = (short int)(dot/sum*scale + 0.5);
      }

      cc = -center;
      {
         k = kernel[center + cc];
         for(c=center;c<cols-center;c++){
            dotx[c] = doty[c+cc] * k;
         }
      }
      while (++cc < center){
         k = kernel[center + cc];
         for(c=center;c<cols-center;c++){
            dotx[c] += doty[c+cc] * k;
         }
      }
      {
         k = kernel[center + cc];
         for(c=center;c<cols-center;c++){
            float dotz = doty[c+cc] * k + dotx[c];
            z0[c] = (short int)(dotz*scale + 0.5);
         }
      }

      for(c=cols-center;c<cols;c++){
         dot = 0.0f;
         sum = 0.0f;
         for(cc=-center;cc<=cols-c;cc++){
               dot += doty[c+cc] * kernel[center+cc];
               sum += kernel[center+cc];
         }
         z0[c] = (short int)(dot/sum*scale + 0.5);
      }

      /****************************************************************************
      * Compute the x-derivative. Adjust the derivative at the borders to avoid
      * losing pixels.
      ****************************************************************************/
      pos = r * cols;
      delta_x[pos] = z0[1] - z0[0];
      pos++;
      for (c = 2; c < cols; c++, pos++) {
          delta_x[pos] = z0[c] - z0[c - 2];
      }
      delta_x[pos] = z0[cols - 1] - z0[cols - 2];

      /****************************************************************************
      * Compute the y-derivative. Adjust the derivative at the borders to avoid
      * losing pixels.
      ****************************************************************************/
      if (r) {
          pos = (r - 1) * cols;
          swap = (r == 1) ? z1 : z2;
          for (c = 0; c < cols; c++, pos++) {
              delta_y[pos] = z0[c] - swap[c];
          }

          /*******************************************************************************
          * PURPOSE: Compute the magnitude of the gradient. This is the square root of
          * the sum of the squared derivative values.
          *******************************************************************************/
          pos = (r - 1) * cols;
          for (c = 15; c < cols; c += 16, pos += 16)
          {
              __m256i vdx = _mm256_loadu_si256((const __m256i*)&delta_x[pos]);
              __m256i vdy = _mm256_loadu_si256((const __m256i*)&delta_y[pos]);

              __m256i vd0 = _mm256_unpacklo_epi16(vdx, vdy);
              __m256i vd1 = _mm256_unpackhi_epi16(vdx, vdy);

              vd0 = _mm256_madd_epi16(vd0, vd0);
              vd1 = _mm256_madd_epi16(vd1, vd1);

              __m256 vf0 = _mm256_cvtepi32_ps(vd0);
              __m256 vf1 = _mm256_cvtepi32_ps(vd1);

              vf0 = _mm256_sqrt_ps(vf0);
              vf1 = _mm256_sqrt_ps(vf1);

              vd0 = _mm256_cvtps_epi32(vf0);
              vd1 = _mm256_cvtps_epi32(vf1);

              __m256i vmag = _mm256_packus_epi32(vd0, vd1);

              _mm256_storeu_si256((__m256i*)&magnitude[pos], vmag);
          }
          for (c -= 15; c < cols; c++, pos++) {
              dx = delta_x[pos];
              dy = delta_y[pos];
              magnitude[pos] = (short)(0.5f + sqrtf(dx * dx + dy * dy));
          }
      }

      swap = z2; z2 = z1; z1 = z0; z0 = swap;
   }

   /****************************************************************************
   * Compute the y-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
   pos = (rows - 1) * cols;
   for (c = 0; c < cols; c++, pos++) {
       delta_y[pos] = z1[c] - z2[c];
   }

   /*******************************************************************************
   * PURPOSE: Compute the magnitude of the gradient. This is the square root of
   * the sum of the squared derivative values.
   *******************************************************************************/
   pos = (rows - 1) * cols;
   for (c = 0; c < cols; c++, pos++) {
       dx = delta_x[pos];
       dy = delta_y[pos];
       magnitude[pos] = (short)(0.5f + sqrtf(dx * dx + dy * dy));
   }

   free(z0);
   free(z1);
   free(z2);
   free(dotx);
   free(doty);
   free(kernel);
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum=0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
   if((*kernel = (float *) malloc((*windowsize) * sizeof(float))) == NULL){
      fprintf(stderr, "Error allocing the gaussian kernel array.\n");
      exit(1);
   }

   for(i=0;i<(*windowsize);i++){
      x = (float)(i - center);
      fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;

   if(VERBOSE){
      printf("The filter coefficients are:\n");
      for(i=0;i<(*windowsize);i++)
         printf("kernel[%d] = %f\n", i, (*kernel)[i]);
   }
}

/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: This routine finds edges that are above some high threshhold or
* are connected to a high pixel by a path of pixels greater than a low
* threshold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, int rows, int cols,
	float tlow, float thigh, unsigned char *edge)
{
   const int delta[8] = { -1, 1, -1 - cols, 0 - cols, 1 - cols, -1 + cols, 0 + cols, 1 + cols };

   int r, pos, numedges, highcount, lowthreshold, highthreshold,
       hist[32768];
   int *poses, *poses_write, e, m, f, i;
   int maximum_mag;

   /****************************************************************************
   * Compute the histogram of the magnitude image. Then use the histogram to
   * compute hysteresis thresholds.
   ****************************************************************************/
   for(r=0;r<32768;r++) hist[r] = 0;

   for (pos = cols, r = cols * (rows - 2); pos < r; pos++) {
       hist[mag[pos]] += (edge[pos] == POSSIBLE_EDGE);
   }

   /****************************************************************************
   * Compute the number of pixels that passed the nonmaximal suppression.
   ****************************************************************************/
   for (maximum_mag = 32768 - 1; maximum_mag && !hist[maximum_mag]; maximum_mag--);

   numedges = 0;
   for (r = 1; r <= maximum_mag; r++) {
       numedges += hist[r];
   }

   highcount = (int)(numedges * thigh + 0.5);

   /****************************************************************************
   * Compute the high threshold value as the (100 * thigh) percentage point
   * in the magnitude of the gradient histogram of all the pixels that passes
   * non-maximal suppression. Then calculate the low threshold as a fraction
   * of the computed high threshold value. John Canny said in his paper
   * "A Computational Approach to Edge Detection" that "The ratio of the
   * high to low threshold in the implementation is in the range two or three
   * to one." That means that in terms of this implementation, we should
   * choose tlow ~= 0.5 or 0.33333.
   ****************************************************************************/
   r = 1;
   numedges = hist[1];
   while((r<(maximum_mag-1)) && (numedges < highcount)){
      r++;
      numedges += hist[r];
   }
   highthreshold = r;
   lowthreshold = (int)(highthreshold * tlow + 0.5);

   if(VERBOSE){
      printf("The input low and high fractions of %f and %f computed to\n",
	 tlow, thigh);
      printf("magnitude of the gradient threshold values of: %d %d\n",
	 lowthreshold, highthreshold);
   }

   /****************************************************************************
   * This loop looks for pixels above the highthreshold to locate edges and
   * then calls follow_edges to continue the edge.
   ****************************************************************************/
   if ((poses = (int*)malloc(sizeof(int) * cols * rows)) == NULL) {
       fprintf(stderr, "Error allocing the wave.\n");
       exit(1);
   }

   for (pos = cols, r = cols * (rows - 2); pos < r; pos++) {
       m = mag[pos];
       e = edge[pos];
       f = (m >= highthreshold) & (e == POSSIBLE_EDGE);
       e = (m > lowthreshold) ? e : NOEDGE;
       edge[pos] = f ? EDGE : e;
   }

   poses_write = poses;
   for (pos = cols, r = cols * (rows - 2); pos < r; pos++) {
       *poses_write = pos;
       poses_write += (edge[pos] == EDGE);
   }

   while (poses_write != poses) {
       r = *--poses_write;

       for (i = 0; i < 8; i++) {
           pos = delta[i] + r;
           *poses_write = pos;

           e = edge[pos];
           f = (e == POSSIBLE_EDGE);
           poses_write += f;

           edge[pos] = f ? EDGE : e;
       }
   }

   free(poses);

   /****************************************************************************
   * Set all the remaining possible edges to non-edges.
   ****************************************************************************/
   for (pos = cols, r = cols * (rows - 2); pos < r; pos++) {
       edge[pos] = (edge[pos] != EDGE) ? NOEDGE : EDGE;
   }
}

/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
    unsigned char *result) 
{
    int rowcount, colcount,count;
    short *magptr, *gxptr, *gyptr;
    short z2L,z2,z2R,z1,m,z3,z4L,z4,z4R,zm;
    short gx,gy,gm,negx,negy,negxy,swapg;
    unsigned char mag1,mag2;
    unsigned char *resultptr;
    

   /****************************************************************************
   * Zero the edges of the result image.
   ****************************************************************************/
    resultptr = result;
    for(count=0; count<ncols; count++){
        *resultptr++ = NOEDGE;
    }

   /****************************************************************************
   * Suppress non-maximum points.
   ****************************************************************************/
   for(rowcount=1,magptr=mag+ncols,gxptr=gradx+ncols,gyptr=grady+ncols;
      rowcount<nrows-2; 
      rowcount++,magptr+=ncols,gxptr+=ncols,gyptr+=ncols,resultptr+=ncols){   
      resultptr[0] = NOEDGE;
      for(colcount=1;colcount<ncols-2;colcount++){   
            z2L = magptr[colcount - ncols - 1];
            z2 = magptr[colcount - ncols];
            z2R = magptr[colcount - ncols + 1];

            z1 = magptr[colcount - 1];
            m = magptr[colcount];
            z3 = magptr[colcount + 1];

            z4L = magptr[colcount + ncols - 1];
            z4 = magptr[colcount + ncols];
            z4R = magptr[colcount + ncols + 1];

            gx = gxptr[colcount];
            gy = gyptr[colcount];

            negx = -(gx < 0);
            gx = (gx < 0) ? -gx : gx;

            negy = -(gy < 0);
            gy = (gy < 0) ? -gy : gy;

            swapg = -(gy > gx);

            //

            zm = (z1 ^ z3) & negx;
            z1 ^= zm;
            z3 ^= zm;

            zm = (z2 ^ z4) & negy;
            z2 ^= zm;
            z4 ^= zm;

            z1 = swapg ? z2 : z1;
            z3 = swapg ? z4 : z3;

            //

            negxy = negx ^ negy;

            z2 = negxy ? z2R : z2L;
            z4 = negxy ? z4L : z4R;

            zm = (z2 ^ z4) & negy;
            z2 ^= zm;
            z4 ^= zm;

            //

            gm = (gx ^ gy) & swapg;
            gx ^= gm;
            gy ^= gm;

            z2 -= z1;
            z4 -= z3;
            z1 -= m;
            z3 -= m;

            mag1 = z2*gy + z1*gx > 0;
            mag2 = z4*gy + z3*gx > -1;

            /* Now determine if the current point is a maximum point */

            resultptr[colcount] = (mag1 | mag2) ? (unsigned char)NOEDGE : (unsigned char)POSSIBLE_EDGE;
      } 
      resultptr[ncols - 2] = NOEDGE;
      resultptr[ncols - 1] = NOEDGE;
    }

   /****************************************************************************
   * Zero the edges of the result image.
   ****************************************************************************/
    for(count=0; count<ncols+ncols; count++) {
        *resultptr++ = NOEDGE;
    }

}
/*******************************************************************************
* FILE: pgm_io.c
* This code was written by Mike Heath. heath@csee.usf.edu (in 1995).
*******************************************************************************/

#include <string.h>

/******************************************************************************
* Function: read_pgm_image
* Purpose: This function reads in an image in PGM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PGM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols)
{
   FILE *fp;
   char buf[71];

   /***************************************************************************
   * Open the input image file for reading if a filename was given. If no
   * filename was provided, set fp to read from standard input.
   ***************************************************************************/
   if(infilename == NULL) fp = stdin;
   else{
      if((fp = fopen(infilename, "rb")) == NULL){
         fprintf(stderr, "Error reading the file %s in read_pgm_image().\n",
            infilename);
         return(0);
      }
   }

   /***************************************************************************
   * Verify that the image is in PGM format, read in the number of columns
   * and rows in the image and scan past all of the header information.
   ***************************************************************************/
   fgets(buf, 70, fp);
   if(strncmp(buf,"P5",2) != 0){
      fprintf(stderr, "The file %s is not in PGM format in ", infilename);
      fprintf(stderr, "read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
   sscanf(buf, "%d %d", cols, rows);
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

   /***************************************************************************
   * Allocate memory to store the image then read the image from the file.
   ***************************************************************************/
   if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
      fprintf(stderr, "Memory allocation failure in read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if((*rows) != fread((*image), (*cols), (*rows), fp)){
      fprintf(stderr, "Error reading the image data in read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      free((*image));
      return(0);
   }

   if(fp != stdin) fclose(fp);
   return(1);
}

/******************************************************************************
* Function: write_pgm_image
* Purpose: This function writes an image in PGM format. The file is either
* written to the file specified by outfilename or to standard output if
* outfilename = NULL. A comment can be written to the header if coment != NULL.
******************************************************************************/
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval)
{
   FILE *fp;

   /***************************************************************************
   * Open the output image file for writing if a filename was given. If no
   * filename was provided, set fp to write to standard output.
   ***************************************************************************/
   if(outfilename == NULL) fp = stdout;
   else{
      if((fp = fopen(outfilename, "wb")) == NULL){
         fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
            outfilename);
         return(0);
      }
   }

   /***************************************************************************
   * Write the header information to the PGM file.
   ***************************************************************************/
   fprintf(fp, "P5\n%d %d\n", cols, rows);
   if(comment != NULL)
      if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);

   /***************************************************************************
   * Write the image data to the file.
   ***************************************************************************/
   if(rows != fwrite(image, cols, rows, fp)){
      fprintf(stderr, "Error writing the image data in write_pgm_image().\n");
      if(fp != stdout) fclose(fp);
      return(0);
   }

   if(fp != stdout) fclose(fp);
   return(1);
}
