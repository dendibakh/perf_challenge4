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
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

int read_pgm_image(char *infilename, unsigned char **image, int *rows,
                   int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
                    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
           float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
                     short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                   short int **magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
                      float tlow, float thigh, unsigned char *edge);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);
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
    FILE *fpdir=NULL;          /* File to write the gradient image to.     */
    unsigned char *nms;        /* Points that are local maximal magnitude. */
    short int *smoothedim,     /* The image after gaussian smoothing.      */
        *delta_x,        /* The first devivative image, x-direction. */
        *delta_y,        /* The first derivative image, y-direction. */
        *magnitude;      /* The magnitude of the gadient image.      */
    int r, c, pos;
    float *dir_radians=NULL;   /* Gradient direction image.                */

    /****************************************************************************
     * Perform gaussian smoothing on the image using the input standard
     * deviation.
     ****************************************************************************/
    if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
    gaussian_smooth(image, rows, cols, sigma, &smoothedim);

    /****************************************************************************
     * Compute the first derivative in the x and y directions.
     ****************************************************************************/
    if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
    derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);

    /****************************************************************************
     * This option to write out the direction of the edge gradient was added
     * to make the information available for computing an edge quality figure
     * of merit.
     ****************************************************************************/
    if(fname != NULL){
        /*************************************************************************
         * Compute the direction up the gradient, in radians that are
         * specified counteclockwise from the positive x-axis.
         *************************************************************************/
        radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

        /*************************************************************************
         * Write the gradient direction image out to a file.
         *************************************************************************/
        if((fpdir = fopen(fname, "wb")) == NULL){
            fprintf(stderr, "Error opening the file %s for writing.\n", fname);
            exit(1);
        }
        fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
        fclose(fpdir);
        free(dir_radians);
    }

    /****************************************************************************
     * Compute the magnitude of the gradient.
     ****************************************************************************/
    if(VERBOSE) printf("Computing the magnitude of the gradient.\n");
    magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

    /****************************************************************************
     * Perform non-maximal suppression.
     ****************************************************************************/
    if(VERBOSE) printf("Doing the non-maximal suppression.\n");
    if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL){
        fprintf(stderr, "Error allocating the nms image.\n");
        exit(1);
    }

    non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

    /****************************************************************************
     * Use hysteresis to mark the edge pixels.
     ****************************************************************************/
    if(VERBOSE) printf("Doing hysteresis thresholding.\n");
    if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL){
        fprintf(stderr, "Error allocating the edge image.\n");
        exit(1);
    }

    apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

    /****************************************************************************
     * Free all of the memory that we allocated except for the edge image that
     * is still being used to store out result.
     ****************************************************************************/
    free(smoothedim);
    free(delta_x);
    free(delta_y);
    free(magnitude);
    free(nms);
}

/*******************************************************************************
 * Procedure: radian_direction
 * Purpose: To compute a direction of the gradient image from component dx and
 * dy images. Because not all derriviatives are computed in the same way, this
 * code allows for dx or dy to have been calculated in different ways.
 *
 * FOR X:  xdirtag = -1  for  [-1 0  1]
 *         xdirtag =  1  for  [ 1 0 -1]
 *
 * FOR Y:  ydirtag = -1  for  [-1 0  1]'
 *         ydirtag =  1  for  [ 1 0 -1]'
 *
 * The resulting angle is in radians measured counterclockwise from the
 * xdirection. The angle points "up the gradient".
 *******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag)
{
    int r, c, pos;
    float *dirim=NULL;
    double dx, dy;

    /****************************************************************************
     * Allocate an image to store the direction of the gradient.
     ****************************************************************************/
    if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        exit(1);
    }
    *dir_radians = dirim;

    for(r=0,pos=0;r<rows;r++){
        for(c=0;c<cols;c++,pos++){
            dx = (double)delta_x[pos];
            dy = (double)delta_y[pos];

            if(xdirtag == 1) dx = -dx;
            if(ydirtag == -1) dy = -dy;

            dirim[pos] = (float)angle_radians(dx, dy);
        }
    }
}

/*******************************************************************************
 * FUNCTION: angle_radians
 * PURPOSE: This procedure computes the angle of a vector with components x and
 * y. It returns this angle in radians with the answer being in the range
 * 0 <= angle <2*PI.
 *******************************************************************************/
double angle_radians(double x, double y)
{
    double xu, yu, ang;

    xu = fabs(x);
    yu = fabs(y);

    if((xu == 0) && (yu == 0)) return(0);

    ang = atan(yu/xu);

    if(x >= 0){
        if(y >= 0) return(ang);
        else return(2*M_PI - ang);
    }
    else{
        if(y >= 0) return(M_PI - ang);
        else return(M_PI + ang);
    }
}

/*******************************************************************************
 * PROCEDURE: magnitude_x_y
 * PURPOSE: Compute the magnitude of the gradient. This is the square root of
 * the sum of the squared derivative values.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                   short int **magnitude)
{
    int r, c, pos, pos2, sq1, sq2;

    /****************************************************************************
     * Allocate an image to store the magnitude of the gradient.
     ****************************************************************************/
    if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the magnitude image.\n");
        exit(1);
    }

    for(r=0,pos=0;r<rows;r++){
        for(c=0;c<cols;c++,pos++){
            sq1 = (int)delta_x[pos] * (int)delta_x[pos];
            sq2 = (int)delta_y[pos] * (int)delta_y[pos];
            (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
        }
    }

}

inline __attribute__((always_inline))
void transpose(__m256i *restrict vec0,
               __m256i *restrict vec1,
               __m256i *restrict vec2,
               __m256i *restrict vec3)
{
    // Transpose the four constitutent 4x4 matrices
    // vecX_Y denotes row X in round Y of transposition

    // f e d c b a 9 8 7 6 5 4 3 2 1 0
    // f e d c b a 9 8 7 6 5 4 3 2 1 0
    // f e d c b a 9 8 7 6 5 4 3 2 1 0
    // f e d c b a 9 8 7 6 5 4 3 2 1 0

    __m256i vec0_0 = _mm256_unpacklo_epi16(*vec0, *vec1);
    __m256i vec1_0 = _mm256_unpackhi_epi16(*vec0, *vec1);
    __m256i vec2_0 = _mm256_unpacklo_epi16(*vec2, *vec3);
    __m256i vec3_0 = _mm256_unpackhi_epi16(*vec2, *vec3);

    // b b a a 9 9 8 8 3 3 2 2 1 1 0 0
    // f f e e d d c c 7 7 6 6 5 5 4 4
    // b b a a 9 9 8 8 3 3 2 2 1 1 0 0
    // f f e e d d c c 7 7 6 6 5 5 4 4

    __m256i vec0_1 = _mm256_unpacklo_epi32(vec0_0, vec2_0);
    __m256i vec1_1 = _mm256_unpacklo_epi32(vec1_0, vec3_0);
    __m256i vec2_1 = _mm256_unpackhi_epi32(vec0_0, vec2_0);
    __m256i vec3_1 = _mm256_unpackhi_epi32(vec1_0, vec3_0);

    // 9 9 9 9 8 8 8 8 1 1 1 1 0 0 0 0
    // d d d d c c c c 5 5 5 5 4 4 4 4
    // b b b b a a a a 3 3 3 3 2 2 2 2
    // f f f f e e e e 7 7 7 7 6 6 6 6 

    *vec0 = _mm256_unpacklo_epi64(vec0_1, vec1_1);
    *vec1 = _mm256_unpackhi_epi64(vec0_1, vec1_1);
    *vec2 = _mm256_unpacklo_epi64(vec2_1, vec3_1);
    *vec3 = _mm256_unpackhi_epi64(vec2_1, vec3_1);

    // c c c c 8 8 8 8 4 4 4 4 0 0 0 0
    // d d d d 9 9 9 9 5 5 5 5 1 1 1 1
    // e e e e a a a a 6 6 6 6 2 2 2 2
    // f f f f b b b b 7 7 7 7 3 3 3 3
}

inline __attribute__((always_inline))
void transpose_f(__m256 *restrict vec0,
                 __m256 *restrict vec1,
                 __m256 *restrict vec2,
                 __m256 *restrict vec3)
{
    __m256 vec0_0 = _mm256_unpacklo_ps(*vec0, *vec1);
    __m256 vec1_0 = _mm256_unpackhi_ps(*vec0, *vec1);
    __m256 vec2_0 = _mm256_unpacklo_ps(*vec2, *vec3);
    __m256 vec3_0 = _mm256_unpackhi_ps(*vec2, *vec3);

    *vec0 = _mm256_unpacklo_pd(vec0_0, vec2_0);
    *vec1 = _mm256_unpackhi_pd(vec0_0, vec2_0);
    *vec2 = _mm256_unpacklo_pd(vec1_0, vec3_0);
    *vec3 = _mm256_unpackhi_pd(vec1_0, vec3_0);
}

/*******************************************************************************
 * PROCEDURE: derrivative_x_y
 * PURPOSE: Compute the first derivative of the image in both the x any y
 * directions. The differential filters that are used are:
 *
 *                                          -1
 *         dx =  -1 0 +1     and       dy =  0
 *                                          +1
 *
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y)
{
    int r, c, pos, pos2;
    short f0, f1, f2, f3;
    /****************************************************************************
     * Allocate images to store the derivatives.
     ****************************************************************************/
    if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }

    /****************************************************************************
     * Compute the x-derivative. Adjust the derivative at the borders to avoid
     * losing pixels.
     ****************************************************************************/
    if(VERBOSE) printf("   Computing the X-direction derivative.\n");

    for(r = 0; r < rows / 4; r++)
    {
        short *in_x = smoothedim + 4 * r * cols;
        short *out_x = (*delta_x) + 4 * r * cols;

        __m256i vec0 = _mm256_loadu_si256((__m256i *)(in_x + 0 * cols));
        __m256i vec1 = _mm256_loadu_si256((__m256i *)(in_x + 1 * cols));
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(in_x + 2 * cols));
        __m256i vec3 = _mm256_loadu_si256((__m256i *)(in_x + 3 * cols));

        transpose(&vec0, &vec1, &vec2, &vec3);
       
        __m256i vec0n = _mm256_loadu_si256((__m256i *)(in_x + 0 * cols + 16));
        __m256i vec1n = _mm256_loadu_si256((__m256i *)(in_x + 1 * cols + 16));
        __m256i vec2n = _mm256_loadu_si256((__m256i *)(in_x + 2 * cols + 16));
        __m256i vec3n = _mm256_loadu_si256((__m256i *)(in_x + 3 * cols + 16));
    
        transpose(&vec0n, &vec1n, &vec2n, &vec3n);

        __m256i vecP = _mm256_permute4x64_epi64(vec3, 0x90);
        __m256i vecN = _mm256_permute4x64_epi64(vec0, 0x39);

        vecP = _mm256_blend_epi32(vecP, vec0, 0x03);
        vecN = _mm256_blend_epi32(vec0n, vecN, 0xc0);

        __m256i r048c = _mm256_sub_epi16(vec1, vecP);
        __m256i r159d = _mm256_sub_epi16(vec2, vec0);
        __m256i r26ae = _mm256_sub_epi16(vec3, vec1);
        __m256i r37bf = _mm256_sub_epi16(vecN, vec2);

        transpose(&r048c, &r159d, &r26ae, &r37bf);
        _mm256_storeu_si256((__m256i *)(out_x + 0 * cols), r048c);
        _mm256_storeu_si256((__m256i *)(out_x + 1 * cols), r159d);
        _mm256_storeu_si256((__m256i *)(out_x + 2 * cols), r26ae);
        _mm256_storeu_si256((__m256i *)(out_x + 3 * cols), r37bf);

        in_x += 32;
        out_x += 16;

        __m256i vecPrev = vec3;
        for(c = 0; c < cols / 16; c++)
        {
            vec0 = vec0n;
            vec1 = vec1n;
            vec2 = vec2n;
            vec3 = vec3n;

            vec0n = _mm256_loadu_si256((__m256i *)(in_x + 0 * cols));
            vec1n = _mm256_loadu_si256((__m256i *)(in_x + 1 * cols));
            vec2n = _mm256_loadu_si256((__m256i *)(in_x + 2 * cols));
            vec3n = _mm256_loadu_si256((__m256i *)(in_x + 3 * cols));

            transpose(&vec0n, &vec1n, &vec2n, &vec3n);

            vecP = _mm256_permute4x64_epi64(vec3, 0x90);
            vecN = _mm256_permute4x64_epi64(vec0, 0x39);
            vecPrev = _mm256_permute4x64_epi64(vecPrev, 0x03); // Put previous highest 64-bits into lowest position
           
            vecP = _mm256_blend_epi32(vecP, vecPrev, 0x03);
            vecN = _mm256_blend_epi32(vec0n, vecN, 0xc0);

            r048c = _mm256_sub_epi16(vec1, vecP);
            r159d = _mm256_sub_epi16(vec2, vec0);
            r26ae = _mm256_sub_epi16(vec3, vec1);
            r37bf = _mm256_sub_epi16(vecN, vec2);

            transpose(&r048c, &r159d, &r26ae, &r37bf);
            _mm256_storeu_si256((__m256i *)(out_x + 0 * cols), r048c);
            _mm256_storeu_si256((__m256i *)(out_x + 1 * cols), r159d);
            _mm256_storeu_si256((__m256i *)(out_x + 2 * cols), r26ae);
            _mm256_storeu_si256((__m256i *)(out_x + 3 * cols), r37bf);

            in_x += 16;
            out_x += 16;
            vecPrev = vec3;
        }
       int tail = cols % 16;
       for(int i = 0; i < 4; i++)
       {
           for(int j = 0; j < tail - 1; j++)
           {
               pos = cols * i + j;
               out_x[pos] = in_x[pos + 1] - in_x[pos - 1];
           }
           pos = cols * i + (tail - 1);
           out_x[pos] = in_x[pos] - in_x[pos - 1];
       }
   }

   int tail_rows = rows % 4;
   short *tail_out_x = *delta_x + tail_rows * cols;
   short *in_x = smoothedim + tail_rows * cols;
   for(r = tail_rows; r < rows; r++)
   {
       *tail_out_x++ = in_x[1] - in_x[0];
       for(c = 1; c < (cols - 1); c++)
       {
           *tail_out_x++ = in_x[c + 1] - in_x[c - 1];
       }
       *tail_out_x++ = in_x[cols - 1] - in_x[cols - 2];
   }

   /****************************************************************************
   * Compute the y-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the Y-direction derivative.\n");
   for(c = 0; c < cols; c++)
       (*delta_y)[c] = smoothedim[c + cols] - smoothedim[c];

   pos = (rows - 1) * cols;
   for(c = 0; c < cols; c++)
       (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos - cols];

   short *r0, *r1, *r2, *r3;
   int tail_c = (cols / 32) * 32;
   short *out_tail_y = *delta_y + tail_c + cols;
   short *in_tail_y = smoothedim + tail_c;
   r0 = in_tail_y;
   r1 = in_tail_y + cols;
   in_tail_y += 2 * cols;
   for(r = 0; r < (rows - 4) / 2 + 1; r++)
   {
       r2 = in_tail_y;
       r3 = in_tail_y + cols;
       for(c = 0; c < cols % 32; c++)
       {
           out_tail_y[c] = r2[c] - r0[c];
           out_tail_y[c + cols] = r3[c] - r0[c];
       }
       r0 = r2;
       r1 = r3;
       out_tail_y += 2 * cols;
       in_tail_y += 2 * cols;
   }

   for(c = 0; c < cols / 32; c++)
   {
       short *out_y = *delta_y + 32 * c + cols;
       short *in_y = smoothedim + 32 * c;

       __m256i vec0_a = _mm256_loadu_si256((__m256i *)(in_y + 0 * cols));
       __m256i vec1_a = _mm256_loadu_si256((__m256i *)(in_y + 1 * cols));
       __m256i vec0_b = _mm256_loadu_si256((__m256i *)(in_y + 0 * cols + 16));
       __m256i vec1_b = _mm256_loadu_si256((__m256i *)(in_y + 1 * cols + 16));
       in_y += 2 * cols;
       for(r = 0; r < (rows - 4) / 2 + 1; r++)
       {
           // Cache line is 64 bytes, which is two AVX2 registers worth
           __m256i vec2_a = _mm256_loadu_si256((__m256i *)(in_y + 0 * cols));
           __m256i vec3_a = _mm256_loadu_si256((__m256i *)(in_y + 1 * cols));
           
           __m256i vec2_b = _mm256_loadu_si256((__m256i *)(in_y + 0 * cols + 16));
           __m256i vec3_b = _mm256_loadu_si256((__m256i *)(in_y + 1 * cols + 16));
           
           __m256i res1_a = _mm256_sub_epi16(vec2_a, vec0_a);
           __m256i res2_a = _mm256_sub_epi16(vec3_a, vec1_a);
           
           __m256i res1_b = _mm256_sub_epi16(vec2_b, vec0_b);
           __m256i res2_b = _mm256_sub_epi16(vec3_b, vec1_b);
           
           vec0_a = vec2_a;
           vec1_a = vec3_a;
           
           vec0_b = vec2_b;
           vec1_b = vec3_b;
           
           _mm256_storeu_si256((__m256i *)(out_y), res1_a);
           _mm256_storeu_si256((__m256i *)(out_y + 16), res1_b);
           _mm256_storeu_si256((__m256i *)(out_y + cols), res2_a);
           _mm256_storeu_si256((__m256i *)(out_y + cols + 16), res2_b);
           out_y += 2 * cols;
           in_y += 2 * cols;
       }
   }
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim)
{
   int r, c, rr, cc,     /* Counter variables. */
      windowsize,        /* Dimension of the gaussian kernel. */
      center;            /* Half of the windowsize. */
   float *tempim,        /* Buffer for separable filter gaussian smoothing. */
         *kernel,        /* A one dimensional gaussian kernel. */
         dot,            /* Dot product summing variable. */
         sum;            /* Sum of the kernel weights variable. */

   /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);
   center = windowsize / 2;

   /****************************************************************************
   * Allocate a temporary buffer image and the smoothed image.
   ****************************************************************************/
   if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
      fprintf(stderr, "Error allocating the buffer image.\n");
      exit(1);
   }
   if(((*smoothedim) = (short int *) calloc(rows*cols, sizeof(short int))) == NULL)
   {
       fprintf(stderr, "Error allocating the smoothed image.\n");
       exit(1);
   }

   if (windowsize == 5)
   {
       __m256 k0 = _mm256_set1_ps(kernel[0]);
       __m256 k1 = _mm256_set1_ps(kernel[1]);
       __m256 k2 = _mm256_set1_ps(kernel[2]);
       __m256 k3 = _mm256_set1_ps(kernel[3]);
       __m256 k4 = _mm256_set1_ps(kernel[4]);
       __m256i vecZero = _mm256_setzero_si256();
       __m256 sumK = _mm256_add_ps(k0, k1);
       sumK = _mm256_add_ps(sumK, k2);
       sumK = _mm256_add_ps(sumK, k3);
       sumK = _mm256_add_ps(sumK, k4);

       __m256i vecPermute = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
       __m256i vecPermute2 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
       __m256i vecBlend = _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff);
       __m256i vecBlend2 = _mm256_set_epi32(0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
       __m256 vecMul = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f);
       __m256 vecMul2 = _mm256_set_ps(0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
       if(VERBOSE) printf("   Blurring the image in the X-direction.\n");
       for(r = 0; r < rows; r++)
       {
           unsigned char *in_x = image + r * cols;
           float *out_x = tempim + r * cols;;
           __m256i vecIn = _mm256_loadu_si256((__m256i *)in_x);
           in_x += 32;
           __m256i vec0_i16 = _mm256_unpacklo_epi8(vecIn, vecZero);
           __m256i vec1_i16 = _mm256_unpackhi_epi8(vecIn, vecZero);

           __m256i vec0_i32 = _mm256_unpacklo_epi16(vec0_i16, vecZero);
           __m256i vec1_i32 = _mm256_unpackhi_epi16(vec0_i16, vecZero);
           __m256i vec2_i32 = _mm256_unpacklo_epi16(vec1_i16, vecZero);
           __m256i vec3_i32 = _mm256_unpackhi_epi16(vec1_i16, vecZero);

           __m256 vec0 = _mm256_cvtepi32_ps(vec0_i32);
           __m256 vec1 = _mm256_cvtepi32_ps(vec1_i32);
           __m256 vec2 = _mm256_cvtepi32_ps(vec2_i32);
           __m256 vec3 = _mm256_cvtepi32_ps(vec3_i32);

           transpose_f(&vec0, &vec1, &vec2, &vec3);
           
           __m256 vecM2 = _mm256_permutevar8x32_ps(vec2, vecPermute);
           __m256 vecM1 = _mm256_permutevar8x32_ps(vec3, vecPermute);
           __m256 k0_special = _mm256_mul_ps(k0, vecMul);
           __m256 k1_special = _mm256_mul_ps(k1, vecMul);
                                                
           __m256 vecM2k0 = _mm256_mul_ps(vecM2, k0_special);
           __m256 vecM1k1 = _mm256_mul_ps(vecM1, k1_special);
           __m256 vec0k2 = _mm256_mul_ps(vec0, k2);
           __m256 vec1k3 = _mm256_mul_ps(vec1, k3);
           __m256 vec2k4 = _mm256_mul_ps(vec2, k4);

           __m256 vecM1k0 = _mm256_mul_ps(vecM1, k0_special);
           __m256 vec0k1 = _mm256_mul_ps(vec0, k1);
           __m256 vec1k2 = _mm256_mul_ps(vec1, k2);
           __m256 vec2k3 = _mm256_mul_ps(vec2, k3);
           __m256 vec3k4 = _mm256_mul_ps(vec3, k4);

           __m256 vec0k0 = _mm256_mul_ps(vec0, k0);
           __m256 vec1k1 = _mm256_mul_ps(vec1, k1);
           __m256 vec2k2 = _mm256_mul_ps(vec2, k2);
           __m256 vec3k3 = _mm256_mul_ps(vec3, k3);

           __m256 vec1k0 = _mm256_mul_ps(vec1, k0);
           __m256 vec2k1 = _mm256_mul_ps(vec2, k1);
           __m256 vec3k2 = _mm256_mul_ps(vec3, k2);

           __m256 vecDot0 = _mm256_add_ps(vecM2k0, vecM1k1);
           vecDot0 = _mm256_add_ps(vecDot0, vec0k2);
           vecDot0 = _mm256_add_ps(vecDot0, vec1k3);
           vecDot0 = _mm256_add_ps(vecDot0, vec2k4);

           __m256 sumk2k3k4 = _mm256_add_ps(k2, k3);
           sumk2k3k4 = _mm256_add_ps(sumk2k3k4, k4);
           
           __m256 vecSum0 = _mm256_add_ps(k0_special, k1_special);
           vecSum0 = _mm256_add_ps(vecSum0, sumk2k3k4);

           __m256 vecDot1 = _mm256_add_ps(vecM1k0, vec0k1);
           vecDot1 = _mm256_add_ps(vecDot1, vec1k2);
           vecDot1 = _mm256_add_ps(vecDot1, vec2k3);
           vecDot1 = _mm256_add_ps(vecDot1, vec3k4);

           __m256 vecSum1 = _mm256_add_ps(k0_special, k1);
           vecSum1 = _mm256_add_ps(vecSum1, sumk2k3k4);

           __m256 vecDot2 = _mm256_add_ps(vec0k0, vec1k1);
           vecDot2 = _mm256_add_ps(vecDot2, vec2k2);
           vecDot2 = _mm256_add_ps(vecDot2, vec3k3);

           __m256 vecDot3 = _mm256_add_ps(vec1k0, vec2k1);
           vecDot3 = _mm256_add_ps(vecDot3, vec3k2);

           __m256 vecRes0 = _mm256_div_ps(vecDot0, vecSum0);
           __m256 vecRes1 = _mm256_div_ps(vecDot1, vecSum1);

           __m256 vecM4 = vec0;
           __m256 vecM3 = vec1;
           vecM2 = vec2;
           vecM1 = vec3;
           for(c = 0; c < (cols / 32) - 1; c++)
           {
               vecIn = _mm256_loadu_si256((__m256i *)in_x);
               in_x += 32;

               vec0_i16 = _mm256_unpacklo_epi8(vecIn, vecZero);
               vec1_i16 = _mm256_unpackhi_epi8(vecIn, vecZero);

               vec0_i32 = _mm256_unpacklo_epi16(vec0_i16, vecZero);
               vec1_i32 = _mm256_unpackhi_epi16(vec0_i16, vecZero);
               vec2_i32 = _mm256_unpacklo_epi16(vec1_i16, vecZero);
               vec3_i32 = _mm256_unpackhi_epi16(vec1_i16, vecZero);

               vec0 = _mm256_cvtepi32_ps(vec0_i32);
               vec1 = _mm256_cvtepi32_ps(vec1_i32);
               vec2 = _mm256_cvtepi32_ps(vec2_i32);
               vec3 = _mm256_cvtepi32_ps(vec3_i32);

               transpose_f(&vec0, &vec1, &vec2, &vec3);

               __m256 vecN1 = _mm256_blendv_ps(vecM4, vec0, vecBlend);
               __m256 vecN2 = _mm256_blendv_ps(vecM3, vec1, vecBlend);
               vecN1 = _mm256_permutevar8x32_ps(vecN1, vecPermute2);
               vecN2 = _mm256_permutevar8x32_ps(vecN2, vecPermute2);
               __m256 vecN1k3 = _mm256_mul_ps(vecN1, k3);
               __m256 vecN1k4 = _mm256_mul_ps(vecN1, k4);
               __m256 vecN2k4 = _mm256_mul_ps(vecN2, k4);
               vecDot2 = _mm256_add_ps(vecDot2, vecN1k4);
               vecDot3 = _mm256_add_ps(vecDot3, vecN1k3);
               vecDot3 = _mm256_add_ps(vecDot3, vecN2k4);
               __m256 vecRes2 = _mm256_div_ps(vecDot2, sumK);
               __m256 vecRes3 = _mm256_div_ps(vecDot3, sumK);
               transpose_f(&vecRes0, &vecRes1, &vecRes2, &vecRes3);

               _mm256_storeu_ps(out_x, vecRes0);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes1);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes2);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes3);
               out_x += 8;

               vecM2 = _mm256_blendv_ps(vec2, vecM2, vecBlend2);
               vecM1 = _mm256_blendv_ps(vec3, vecM1, vecBlend2);
               vecM2 = _mm256_permutevar8x32_ps(vecM2, vecPermute);
               vecM1 = _mm256_permutevar8x32_ps(vecM1, vecPermute);

               vecM2k0 = _mm256_mul_ps(vecM2, k0);
               vecM1k1 = _mm256_mul_ps(vecM1, k1);
               vec0k2 = _mm256_mul_ps(vec0, k2);
               vec1k3 = _mm256_mul_ps(vec1, k3);
               vec2k4 = _mm256_mul_ps(vec2, k4);
               
               vecM1k0 = _mm256_mul_ps(vecM1, k0);
               vec0k1 = _mm256_mul_ps(vec0, k1);
               vec1k2 = _mm256_mul_ps(vec1, k2);
               vec2k3 = _mm256_mul_ps(vec2, k3);
               vec3k4 = _mm256_mul_ps(vec3, k4);

               vec0k0 = _mm256_mul_ps(vec0, k0);
               vec1k1 = _mm256_mul_ps(vec1, k1);
               vec2k2 = _mm256_mul_ps(vec2, k2);
               vec3k3 = _mm256_mul_ps(vec3, k3);

               vec1k0 = _mm256_mul_ps(vec1, k0);
               vec2k1 = _mm256_mul_ps(vec2, k1);
               vec3k2 = _mm256_mul_ps(vec3, k2);

               vecDot0 = _mm256_add_ps(vecM2k0, vecM1k1);
               vecDot0 = _mm256_add_ps(vecDot0, vec0k2);
               vecDot0 = _mm256_add_ps(vecDot0, vec1k3);

               vecDot1 = _mm256_add_ps(vecM1k0, vec0k1);
               vecDot1 = _mm256_add_ps(vecDot1, vec1k2);
               vecDot1 = _mm256_add_ps(vecDot1, vec2k3);

               vecDot2 = _mm256_add_ps(vec0k0, vec1k1);
               vecDot2 = _mm256_add_ps(vecDot2, vec2k2);
               vecDot2 = _mm256_add_ps(vecDot2, vec3k3);

               vecDot3 = _mm256_add_ps(vec1k0, vec2k1);
               vecDot3 = _mm256_add_ps(vecDot3, vec3k2);

               vecRes0 = _mm256_div_ps(vecDot0, sumK);
               vecRes1 = _mm256_div_ps(vecDot1, sumK);

               vecM4 = vec0;
               vecM3 = vec1;
               vecM2 = vec2;
               vecM1 = vec3;
           }
           // Tail
           {
               vecIn = _mm256_loadu_si256((__m256i *)in_x);
               in_x += 32;

               vec0_i16 = _mm256_unpacklo_epi8(vecIn, vecZero);
               vec1_i16 = _mm256_unpackhi_epi8(vecIn, vecZero);

               vec0_i32 = _mm256_unpacklo_epi16(vec0_i16, vecZero);
               vec1_i32 = _mm256_unpackhi_epi16(vec0_i16, vecZero);
               vec2_i32 = _mm256_unpacklo_epi16(vec1_i16, vecZero);
               vec3_i32 = _mm256_unpackhi_epi16(vec1_i16, vecZero);

               vec0 = _mm256_cvtepi32_ps(vec0_i32);
               vec1 = _mm256_cvtepi32_ps(vec1_i32);
               vec2 = _mm256_cvtepi32_ps(vec2_i32);
               vec3 = _mm256_cvtepi32_ps(vec3_i32);

               transpose_f(&vec0, &vec1, &vec2, &vec3);

               __m256 vecN1 = _mm256_blendv_ps(vecM4, vec0, vecBlend);
               __m256 vecN2 = _mm256_blendv_ps(vecM3, vec1, vecBlend);
               vecN1 = _mm256_permutevar8x32_ps(vecN1, vecPermute2);
               vecN2 = _mm256_permutevar8x32_ps(vecN2, vecPermute2);
               __m256 vecN1k3 = _mm256_mul_ps(vecN1, k3);
               __m256 vecN1k4 = _mm256_mul_ps(vecN1, k4);
               __m256 vecN2k4 = _mm256_mul_ps(vecN2, k4);
               vecDot2 = _mm256_add_ps(vecDot2, vecN1k4);
               vecDot3 = _mm256_add_ps(vecDot3, vecN1k3);
               vecDot3 = _mm256_add_ps(vecDot3, vecN2k4);
               __m256 vecRes2 = _mm256_div_ps(vecDot2, sumK);
               __m256 vecRes3 = _mm256_div_ps(vecDot3, sumK);
               transpose_f(&vecRes0, &vecRes1, &vecRes2, &vecRes3);

               _mm256_storeu_ps(out_x, vecRes0);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes1);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes2);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes3);
               out_x += 8;

               __m256 k3_special = _mm256_mul_ps(vecMul2, k3);
               __m256 k4_special = _mm256_mul_ps(vecMul2, k4);
               vecN1 = _mm256_permutevar8x32_ps(vec0, vecPermute2);
               vecN2 = _mm256_permutevar8x32_ps(vec1, vecPermute2);

               vecM2k0 = _mm256_mul_ps(vecM2, k0);
               vecM1k1 = _mm256_mul_ps(vecM1, k1);
               vec0k2 = _mm256_mul_ps(vec0, k2);
               vec1k3 = _mm256_mul_ps(vec1, k3);
               vec2k4 = _mm256_mul_ps(vec2, k4);

               vecM1k0 = _mm256_mul_ps(vecM1, k0);
               vec0k1 = _mm256_mul_ps(vec0, k1);
               vec1k2 = _mm256_mul_ps(vec1, k2);
               vec2k3 = _mm256_mul_ps(vec2, k3);
               vec3k4 = _mm256_mul_ps(vec3, k4);

               vec0k0 = _mm256_mul_ps(vec0, k0);
               vec1k1 = _mm256_mul_ps(vec1, k1);
               vec2k2 = _mm256_mul_ps(vec2, k2);
               vec3k3 = _mm256_mul_ps(vec3, k3);
               vecN1k4 = _mm256_mul_ps(vecN1, k4_special);

               vec1k0 = _mm256_mul_ps(vec1, k0);
               vec2k1 = _mm256_mul_ps(vec2, k1);
               vec3k2 = _mm256_mul_ps(vec3, k2);
               vecN1k3 = _mm256_mul_ps(vecN1, k3_special);
               vecN2k4 = _mm256_mul_ps(vecN2, k4_special);

               vecDot0 = _mm256_add_ps(vecM2k0, vecM1k1);
               vecDot0 = _mm256_add_ps(vecDot0, vec0k2);
               vecDot0 = _mm256_add_ps(vecDot0, vec1k3);
               vecDot0 = _mm256_add_ps(vecDot0, vec2k4);

               vecDot1 = _mm256_add_ps(vecM1k0, vec0k1);
               vecDot1 = _mm256_add_ps(vecDot1, vec1k2);
               vecDot1 = _mm256_add_ps(vecDot1, vec2k3);
               vecDot2 = _mm256_add_ps(vecDot1, vec3k4);

               vecDot2 = _mm256_add_ps(vec0k0, vec1k1);
               vecDot2 = _mm256_add_ps(vecDot2, vec2k2);
               vecDot2 = _mm256_add_ps(vecDot2, vec3k3);
               vecDot2 = _mm256_add_ps(vecDot2, vecN1k4);

               vecDot3 = _mm256_add_ps(vec1k0, vec2k1);
               vecDot3 = _mm256_add_ps(vecDot3, vec3k2);
               vecDot3 = _mm256_add_ps(vecDot3, vecN1k3);
               vecDot3 = _mm256_add_ps(vecDot3, vecN2k4);
               
               __m256 vecSumk0k1k2 = _mm256_add_ps(k0, k1);
               vecSumk0k1k2 = _mm256_add_ps(vecSumk0k1k2, k2);

               __m256 vecSum2 = _mm256_add_ps(vecSumk0k1k2, k3);
               vecSum2 = _mm256_add_ps(vecSumk0k1k2, k4_special);
               
               __m256 vecSum3 = _mm256_add_ps(vecSumk0k1k2, k3_special);
               vecSum3 = _mm256_add_ps(vecSum3, k4_special);

               vecRes0 = _mm256_div_ps(vecDot0, sumK);
               vecRes1 = _mm256_div_ps(vecDot1, sumK);
               vecRes2 = _mm256_div_ps(vecDot2, vecSum2);
               vecRes3 = _mm256_div_ps(vecDot3, vecSum3);

               _mm256_storeu_ps(out_x, vecRes0);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes1);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes2);
               out_x += 8;
               _mm256_storeu_ps(out_x, vecRes3);
               out_x += 8;
           }
       }

       if(VERBOSE) printf("   Bluring the image in the Y-direction.\n");
       __m256 sumk2k3k4 = _mm256_add_ps(k2, k3);
       sumk2k3k4 = _mm256_add_ps(sumk2k3k4, k4);
       __m256 sumk1k2k3k4 = _mm256_add_ps(sumk2k3k4, k1);

       __m256 sumk0k1k2 = _mm256_add_ps(k0, k1);
       sumk0k1k2 = _mm256_add_ps(sumk0k1k2, k2);
       __m256 sumk0k1k2k3 = _mm256_add_ps(sumk0k1k2, k3);
       
       __m256 vecBoost = _mm256_set1_ps(BOOSTBLURFACTOR);
       __m256i vecShuffle = _mm256_set_epi8(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00);
       for(int c = 0; c < cols / 16; c++)
       {
           float *in_x = tempim + 16 * c;
           short *out_x = (*smoothedim) + 16 * c;
           __m256 vec0_0 = _mm256_loadu_ps(in_x);
           __m256 vec1_0 = _mm256_loadu_ps(in_x + cols);
           __m256 vec2_0 = _mm256_loadu_ps(in_x + cols * 2);

           __m256 vec0k2_0 = _mm256_mul_ps(vec0_0, k2);
           __m256 vec1k3_0 = _mm256_mul_ps(vec1_0, k3);
           __m256 vec2k4_0 = _mm256_mul_ps(vec2_0, k4);

           __m256 vecDot0_0 = _mm256_add_ps(vec0k2_0, vec1k3_0);
           vecDot0_0 = _mm256_add_ps(vecDot0_0, vec2k4_0);

           __m256 vec0_1 = _mm256_loadu_ps(in_x + 8);
           __m256 vec1_1 = _mm256_loadu_ps(in_x + cols + 8);
           __m256 vec2_1 = _mm256_loadu_ps(in_x + cols * 2 + 8);

           __m256 vec0k2_1 = _mm256_mul_ps(vec0_1, k2);
           __m256 vec1k3_1 = _mm256_mul_ps(vec1_1, k3);
           __m256 vec2k4_1 = _mm256_mul_ps(vec2_1, k4);

           __m256 vecDot0_1 = _mm256_add_ps(vec0k2_1, vec1k3_1);
           vecDot0_1 = _mm256_add_ps(vecDot0_1, vec2k4_1);

           __m256 vec3_0 = _mm256_loadu_ps(in_x + cols * 3);
           __m256 vec3_1 = _mm256_loadu_ps(in_x + cols * 3 + 8);

           __m256 vec0k1_0 = _mm256_mul_ps(vec0_0, k1);
           __m256 vec1k2_0 = _mm256_mul_ps(vec1_0, k2);
           __m256 vec2k3_0 = _mm256_mul_ps(vec2_0, k3);
           __m256 vec3k4_0 = _mm256_mul_ps(vec3_0, k4);

           __m256 vec0k1_1 = _mm256_mul_ps(vec0_1, k1);
           __m256 vec1k2_1 = _mm256_mul_ps(vec1_1, k2);
           __m256 vec2k3_1 = _mm256_mul_ps(vec2_1, k3);
           __m256 vec3k4_1 = _mm256_mul_ps(vec3_1, k4);

           __m256 vecDot1_0 = _mm256_add_ps(vec0k1_0, vec1k2_0);
           vecDot1_0 = _mm256_add_ps(vecDot1_0, vec2k3_0);
           vecDot1_0 = _mm256_add_ps(vecDot1_0, vec3k4_0);

           __m256 vecDot1_1 = _mm256_add_ps(vec0k1_1, vec1k2_1);
           vecDot1_1 = _mm256_add_ps(vecDot1_1, vec2k3_1);
           vecDot1_1 = _mm256_add_ps(vecDot1_1, vec3k4_1);

           __m256 vecRes0_0 = _mm256_mul_ps(vecDot0_0, vecBoost);
           __m256 vecRes0_1 = _mm256_mul_ps(vecDot0_1, vecBoost);
           __m256 vecRes1_0 = _mm256_mul_ps(vecDot1_0, vecBoost);
           __m256 vecRes1_1 = _mm256_mul_ps(vecDot1_1, vecBoost);
           vecRes0_0 = _mm256_div_ps(vecRes0_0, sumk2k3k4);
           vecRes0_1 = _mm256_div_ps(vecRes0_1, sumk2k3k4);
           vecRes1_0 = _mm256_div_ps(vecRes1_0, sumk1k2k3k4);
           vecRes1_1 = _mm256_div_ps(vecRes1_1, sumk1k2k3k4);
           __m256i vecRes0_0i = _mm256_cvtps_epi32(vecRes0_0);
           __m256i vecRes0_1i = _mm256_cvtps_epi32(vecRes0_1);
           __m256i vecRes1_0i = _mm256_cvtps_epi32(vecRes1_0);
           __m256i vecRes1_1i = _mm256_cvtps_epi32(vecRes1_1);

           vecRes0_0i = _mm256_shuffle_epi8(vecRes0_0i, vecShuffle);
           vecRes0_1i = _mm256_shuffle_epi8(vecRes0_1i, vecShuffle);
           vecRes1_0i = _mm256_shuffle_epi8(vecRes1_0i, vecShuffle);
           vecRes1_1i = _mm256_shuffle_epi8(vecRes1_1i, vecShuffle);

           vecRes0_0i = _mm256_permute4x64_epi64(vecRes0_0i, 0x08);
           vecRes0_1i = _mm256_permute4x64_epi64(vecRes0_0i, 0x08);
           vecRes1_0i = _mm256_permute4x64_epi64(vecRes0_0i, 0x08);
           vecRes1_1i = _mm256_permute4x64_epi64(vecRes0_0i, 0x08);

           __m256i vecRes0 = _mm256_permute2x128_si256(vecRes0_0i, vecRes0_1i, 0x20);
           __m256i vecRes1 = _mm256_permute2x128_si256(vecRes1_0i, vecRes1_1i, 0x20);
           _mm256_storeu_si256((__m256i *)out_x, vecRes0);
           _mm256_storeu_si256((__m256i *)(out_x + cols), vecRes0);
           out_x += 2 * cols;

           __m256 vecM2_0 = vec0_0;
           __m256 vecM2_1 = vec0_1;
           __m256 vecM1_0 = vec1_0;
           __m256 vecM1_1 = vec1_1;
           vec0_0 = vec2_0;
           vec0_1 = vec2_1;
           vec1_0 = vec3_0;
           vec1_1 = vec3_1;
           in_x += 4 * cols;
           for(r = 2; r < rows - 2; r++)
           {
               vec2_0 = _mm256_loadu_ps(in_x);
               vec2_1 = _mm256_loadu_ps(in_x + 8);
               in_x += cols;

               __m256 vecM2k0_0 = _mm256_mul_ps(vecM2_0, k0);
               __m256 vecM1k1_0 = _mm256_mul_ps(vecM1_0, k1);
               __m256 vec0k2_0 = _mm256_mul_ps(vec0_0, k2);
               __m256 vec1k3_0 = _mm256_mul_ps(vec1_0, k3);
               __m256 vec2k4_0 = _mm256_mul_ps(vec2_0, k4);

               __m256 vecM2k0_1 = _mm256_mul_ps(vecM2_1, k0);
               __m256 vecM1k1_1 = _mm256_mul_ps(vecM1_1, k1);
               __m256 vec0k2_1 = _mm256_mul_ps(vec0_1, k2);
               __m256 vec1k3_1 = _mm256_mul_ps(vec1_1, k3);
               __m256 vec2k4_1 = _mm256_mul_ps(vec2_1, k4);

               __m256 vecDot_0 = _mm256_add_ps(vecM2k0_0, vecM1k1_0);
               _mm256_add_ps(vecDot_0, vec0k2_0);
               _mm256_add_ps(vecDot_0, vec1k3_0);
               _mm256_add_ps(vecDot_0, vec2k4_0);

               __m256 vecDot_1 = _mm256_add_ps(vecM2k0_1, vecM1k1_1);
               _mm256_add_ps(vecDot_1, vec0k2_1);
               _mm256_add_ps(vecDot_1, vec1k3_1);
               _mm256_add_ps(vecDot_1, vec2k4_1);

               __m256 vecRes_0 = _mm256_mul_ps(vecDot_0, vecBoost);
               __m256 vecRes_1 = _mm256_mul_ps(vecDot_1, vecBoost);
               vecRes_0 = _mm256_div_ps(vecRes_0, sumK);
               vecRes_1 = _mm256_div_ps(vecRes_1, sumK);
               __m256i vecRes_0i = _mm256_cvtps_epi32(vecRes_0);
               __m256i vecRes_1i = _mm256_cvtps_epi32(vecRes_1);
               vecRes_0i = _mm256_shuffle_epi8(vecRes_0i, vecShuffle);
               vecRes_1i = _mm256_shuffle_epi8(vecRes_1i, vecShuffle);

               vecRes_0i = _mm256_permute4x64_epi64(vecRes_0i, 0x08);
               vecRes_1i = _mm256_permute4x64_epi64(vecRes_0i, 0x08);
               __m256i vecRes = _mm256_permute2x128_si256(vecRes_0i, vecRes_1i, 0x20);
               _mm256_storeu_si256((__m256i *)out_x, vecRes);
               out_x += cols;

               vecM2_0 = vecM1_0;
               vecM2_1 = vecM1_1;
               vecM1_0 = vec0_0;
               vecM1_1 = vec1_1;
               vec0_0 = vec1_0;
               vec0_1 = vec1_1;
               vec1_0 = vec2_0;
               vec1_1 = vec2_1;
           }
           {
               __m256 vec2_0 = _mm256_loadu_ps(in_x);
               __m256 vec2_1 = _mm256_loadu_ps(in_x + 8);

               __m256 vecM2k0_0 = _mm256_mul_ps(vecM2_0, k0);
               __m256 vecM1k1_0 = _mm256_mul_ps(vecM1_0, k1);
               __m256 vec0k2_0 = _mm256_mul_ps(vec0_0, k2);
               __m256 vec1k3_0 = _mm256_mul_ps(vec1_0, k3);
               __m256 vec2k4_0 = _mm256_mul_ps(vec2_0, k4);

               __m256 vecM2k0_1 = _mm256_mul_ps(vecM2_1, k0);
               __m256 vecM1k1_1 = _mm256_mul_ps(vecM1_1, k1);
               __m256 vec0k2_1 = _mm256_mul_ps(vec0_1, k2);
               __m256 vec1k3_1 = _mm256_mul_ps(vec1_1, k3);
               __m256 vec2k4_1 = _mm256_mul_ps(vec2_1, k4);

               __m256 vecDot_0 = _mm256_add_ps(vecM2k0_0, vecM1k1_0);
               _mm256_add_ps(vecDot_0, vec0k2_0);
               _mm256_add_ps(vecDot_0, vec1k3_0);
               _mm256_add_ps(vecDot_0, vec2k4_0);

               __m256 vecDot_1 = _mm256_add_ps(vecM2k0_1, vecM1k1_1);
               _mm256_add_ps(vecDot_1, vec0k2_1);
               _mm256_add_ps(vecDot_1, vec1k3_1);
               _mm256_add_ps(vecDot_1, vec2k4_1);

               __m256 vecRes_0 = _mm256_mul_ps(vecDot_0, vecBoost);
               __m256 vecRes_1 = _mm256_mul_ps(vecDot_1, vecBoost);
               vecRes_0 = _mm256_div_ps(vecRes_0, sumK);
               vecRes_1 = _mm256_div_ps(vecRes_1, sumK);
               __m256i vecRes_0i = _mm256_cvtps_epi32(vecRes_0);
               __m256i vecRes_1i = _mm256_cvtps_epi32(vecRes_1);
               vecRes_0i = _mm256_shuffle_epi8(vecRes_0i, vecShuffle);
               vecRes_1i = _mm256_shuffle_epi8(vecRes_1i, vecShuffle);

               vecRes_0i = _mm256_permute4x64_epi64(vecRes_0i, 0x08);
               vecRes_1i = _mm256_permute4x64_epi64(vecRes_0i, 0x08);
               __m256i vecRes = _mm256_permute2x128_si256(vecRes_0i, vecRes_1i, 0x20);
               _mm256_storeu_si256((__m256i *)out_x, vecRes);
               out_x += cols;

               // Second to last element
               __m256 vecM1k0_0 = _mm256_mul_ps(vecM1_0, k0);
               __m256 vec0k1_0 = _mm256_mul_ps(vec0_0, k1);
               __m256 vec1k2_0 = _mm256_mul_ps(vec1_0, k2);
               __m256 vec2k3_0 = _mm256_mul_ps(vec2_0, k3);

               __m256 vecM1k0_1 = _mm256_mul_ps(vecM1_1, k0);
               __m256 vec0k1_1 = _mm256_mul_ps(vec0_1, k1);
               __m256 vec1k2_1 = _mm256_mul_ps(vec1_1, k2);
               __m256 vec2k3_1 = _mm256_mul_ps(vec2_1, k3);

               __m256 vecDot1_0 = _mm256_add_ps(vecM1k0_0, vec0k1_0);
               _mm256_add_ps(vecDot1_0, vec1k2_0);
               _mm256_add_ps(vecDot1_0, vec2k3_0);
               _mm256_add_ps(vecDot1_0, vec3k4_0);

               __m256 vecDot1_1 = _mm256_add_ps(vecM1k0_1, vec0k1_1);
               _mm256_add_ps(vecDot1_1, vec1k2_1);
               _mm256_add_ps(vecDot1_1, vec2k3_1);
               _mm256_add_ps(vecDot1_1, vec3k4_1);

               __m256 vecRes1_0 = _mm256_mul_ps(vecDot1_0, vecBoost);
               __m256 vecRes1_1 = _mm256_mul_ps(vecDot1_1, vecBoost);
               vecRes1_0 = _mm256_div_ps(vecRes1_0, sumk0k1k2k3);
               vecRes1_1 = _mm256_div_ps(vecRes1_1, sumk0k1k2k3);
               __m256i vecRes1_0i = _mm256_cvtps_epi32(vecRes1_0);
               __m256i vecRes1_1i = _mm256_cvtps_epi32(vecRes1_1);
               vecRes1_0i = _mm256_shuffle_epi8(vecRes1_0i, vecShuffle);
               vecRes1_1i = _mm256_shuffle_epi8(vecRes1_1i, vecShuffle);

               vecRes1_0i = _mm256_permute4x64_epi64(vecRes1_0i, 0x08);
               vecRes1_1i = _mm256_permute4x64_epi64(vecRes1_0i, 0x08);
               __m256i vecRes1 = _mm256_permute2x128_si256(vecRes1_0i, vecRes1_1i, 0x20);
               _mm256_storeu_si256((__m256i *)out_x, vecRes1);
               out_x += cols;

               // Last element
               __m256 vec0k0_0 = _mm256_mul_ps(vec0_0, k0);
               __m256 vec1k1_0 = _mm256_mul_ps(vec1_0, k1);
               __m256 vec2k2_0 = _mm256_mul_ps(vec2_0, k2);

               __m256 vec0k0_1 = _mm256_mul_ps(vec0_1, k0);
               __m256 vec1k1_1 = _mm256_mul_ps(vec1_1, k1);
               __m256 vec2k2_1 = _mm256_mul_ps(vec2_1, k2);

               __m256 vecDot2_0 = _mm256_add_ps(vec0k0_0, vec1k1_0);
               _mm256_add_ps(vecDot2_0, vec2k2_0);

               __m256 vecDot2_1 = _mm256_add_ps(vec0k0_1, vec1k1_1);
               _mm256_add_ps(vecDot2_1, vec2k2_1);

               __m256 vecRes2_0 = _mm256_mul_ps(vecDot2_0, vecBoost);
               __m256 vecRes2_1 = _mm256_mul_ps(vecDot2_1, vecBoost);
               vecRes2_0 = _mm256_div_ps(vecRes2_0, sumk0k1k2);
               vecRes2_1 = _mm256_div_ps(vecRes2_1, sumk0k1k2);
               __m256i vecRes2_0i = _mm256_cvtps_epi32(vecRes2_0);
               __m256i vecRes2_1i = _mm256_cvtps_epi32(vecRes2_1);
               vecRes2_0i = _mm256_shuffle_epi8(vecRes2_0i, vecShuffle);
               vecRes2_1i = _mm256_shuffle_epi8(vecRes2_1i, vecShuffle);

               vecRes2_0i = _mm256_permute4x64_epi64(vecRes2_0i, 0x08);
               vecRes2_1i = _mm256_permute4x64_epi64(vecRes2_0i, 0x08);
               __m256i vecRes2 = _mm256_permute2x128_si256(vecRes2_0i, vecRes2_1i, 0x20);
               _mm256_storeu_si256((__m256i *)out_x, vecRes2);
           }
       }
       return;
   }

   /****************************************************************************
   * Blur in the x - direction.
   ****************************************************************************/
   if(VERBOSE) printf("   Bluring the image in the X-direction.\n");
   for(r = 0; r < rows; r++)
   {
       for(c = 0; c < cols; c++)
       {
           dot = 0.0;
           sum = 0.0;
           for(cc = (-center); cc <= center; cc++)
           {
               if(((c+cc) >= 0) && ((c+cc) < cols))
               {
                   dot += (float)image[r*cols + (c+cc)] * kernel[center + cc];
                   sum += kernel[center + cc];
               }
           }
           tempim[r*cols+c] = dot/sum;
       }
   }

   /****************************************************************************
   * Blur in the y - direction.
   ****************************************************************************/
   if(VERBOSE) printf("   Bluring the image in the Y-direction.\n");
   r = 0;
   for(int bundleR = 0; bundleR < rows / 4; bundleR++)
   {
       c = 0;
       for(int bundleC = 0; bundleC < cols / 16; bundleC++)
       {
           r = bundleR * 4;
           for(int ir = 0; ir < 4; ir++)
           {
               c = bundleC * 16;
               for(int ic = 0; ic < 16; ic++)
               {
                   sum = 0.0;
                   dot = 0.0;
                   for(rr = -center; rr <= center; rr++)
                   {
                       if((r + rr) >= 0 && (r + rr) < rows)
                       {
                           dot += tempim[(r + rr) * cols + c] * kernel[center + rr];
                           sum += kernel[center + rr];
                       }
                   }
                   (*smoothedim)[r * cols + c] = (short)(dot * BOOSTBLURFACTOR / sum + 0.5);
                   c++;
               }
               r++;
           }
       }
   }
   free(tempim);
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
   if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL){
      fprintf(stderr, "Error callocing the gaussian kernel array.\n");
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

#include <stdio.h>
#include <stdlib.h>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

/*******************************************************************************
* PROCEDURE: follow_edges
* PURPOSE: This procedure edges is a recursive routine that traces edgs along
* all paths whose magnitude values remain above some specifyable lower
* threshhold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
   int cols)
{
   short *tempmagptr;
   unsigned char *tempmapptr;
   int i;
   float thethresh;
   int x[8] = {1,1,0,-1,-1,-1,0,1},
       y[8] = {0,1,1,1,0,-1,-1,-1};

   for(i=0;i<8;i++){
      tempmapptr = edgemapptr - y[i]*cols + x[i];
      tempmagptr = edgemagptr - y[i]*cols + x[i];

      if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)){
         *tempmapptr = (unsigned char) EDGE;
         follow_edges(tempmapptr,tempmagptr, lowval, cols);
      }
   }
}

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: This routine finds edges that are above some high threshhold or
* are connected to a high pixel by a path of pixels greater than a low
* threshold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
	float tlow, float thigh, unsigned char *edge)
{
   int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
       i, hist[32768], rr, cc;
   short int maximum_mag, sumpix;

   /****************************************************************************
   * Initialize the edge map to possible edges everywhere the non-maximal
   * suppression suggested there could be an edge except for the border. At
   * the border we say there can not be an edge because it makes the
   * follow_edges algorithm more efficient to not worry about tracking an
   * edge off the side of the image.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
	 if(nms[pos] == POSSIBLE_EDGE) edge[pos] = POSSIBLE_EDGE;
	 else edge[pos] = NOEDGE;
      }
   }

   for(r=0,pos=0;r<rows;r++,pos+=cols){
      edge[pos] = NOEDGE;
      edge[pos+cols-1] = NOEDGE;
   }
   pos = (rows-1) * cols;
   for(c=0;c<cols;c++,pos++){
      edge[c] = NOEDGE;
      edge[pos] = NOEDGE;
   }

   /****************************************************************************
   * Compute the histogram of the magnitude image. Then use the histogram to
   * compute hysteresis thresholds.
   ****************************************************************************/
   for(r = 0; r < 32768; r++)
       hist[r] = 0;
   for(r = 0, pos = 0; r < rows; r++)
   {
       for(c=0;c<cols;c++,pos++)
           if(edge[pos] == POSSIBLE_EDGE)
               hist[mag[pos]]++;
   }

   /****************************************************************************
   * Compute the number of pixels that passed the nonmaximal suppression.
   ****************************************************************************/
   for(r=1,numedges=0;r<32768;r++){
      if(hist[r] != 0) maximum_mag = r;
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
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
	 if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)){
            edge[pos] = EDGE;
            follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
	 }
      }
   }

   /****************************************************************************
   * Set all the remaining possible edges to non-edges.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] != EDGE) edge[pos] = NOEDGE;
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
    short *magrowptr,*magptr;
    short *gxrowptr,*gxptr;
    short *gyrowptr,*gyptr,z1,z2;
    short m00,gx,gy;
    float mag1,mag2,xperp,yperp;
    unsigned char *resultrowptr, *resultptr;
    
    /****************************************************************************
     * Zero the edges of the result image.
     ****************************************************************************/
    resultrowptr = result;
    resultptr = result + ncols * (nrows - 1);    
    for(count = 0; count < ncols; count++)
    {
        *resultptr++ = 0;
        *resultrowptr++ = 0;
    }

    resultptr = result;
    resultrowptr = result + ncols - 1;
    for(count=0; count < nrows; count++)
    {
        *resultptr = 0;
        *resultrowptr = 0;
        resultptr += ncols;
        resultrowptr += ncols;
    }

    /****************************************************************************
     * Suppress non-maximum points.
     ****************************************************************************/
    magrowptr = mag + ncols;
    gxrowptr = gradx + ncols;
    gyrowptr = grady + ncols;
    resultrowptr = result + ncols + 1;
    for(rowcount = 1; rowcount < nrows - 2; rowcount++)
    {
        magptr = magrowptr;
        gxptr = gxrowptr;
        gyptr = gyrowptr;
        resultptr = resultrowptr;

        for(colcount = 1; colcount < ncols - 2; colcount++)
        {
            m00 = *magptr;
            if(m00 == 0)
            {
                *resultptr = (unsigned char) NOEDGE;
            }
            else
            {
                gx = *gxptr;
                gy = *gyptr;
                xperp = -gx / ((float)m00);
                yperp = gy / ((float)m00);
            }

            if(gx >= 0)
            {
                if(gy >= 0)
                {
                    if(gx >= gy)
                    {
                        /* 111 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);
                        
                        mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                        
                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {    
                        /* 110 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp; 
                    }
                }
                else
                {
                    if (gx >= -gy)
                    {
                        /* 101 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;
            
                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {    
                        /* 100 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp; 
                    }
                }
            }
            else
            {
                if ((gy = *gyptr) >= 0)
                {
                    if (-gx >= gy)
                    {          
                        /* 011 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {
                        /* 010 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                    }
                }
                else
                {
                    if (-gx > -gy)
                    {
                        /* 001 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {
                        /* 000 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                    }
                }
            } 

            /* Now determine if the current point is a maximum point */

            if ((mag1 > 0.0) || (mag2 > 0.0))
            {
                *resultptr = (unsigned char) NOEDGE;
            }
            else
            {    
                if (mag2 == 0.0)
                    *resultptr = (unsigned char) NOEDGE;
                else
                    *resultptr = (unsigned char) POSSIBLE_EDGE;
            }
            magptr++;
            gxptr++;
            gyptr++;
            resultptr++;
        }
        magrowptr += ncols;
        gyrowptr += ncols;
        gxrowptr += ncols;
        resultrowptr += ncols;
    }
}
/*******************************************************************************
* FILE: pgm_io.c
* This code was written by Mike Heath. heath@csee.usf.edu (in 1995).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
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
      if((fp = fopen(outfilename, "w")) == NULL){
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

/******************************************************************************
* Function: read_ppm_image
* Purpose: This function reads in an image in PPM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PPM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_ppm_image(char *infilename, unsigned char **image_red, 
    unsigned char **image_grn, unsigned char **image_blu, int *rows,
    int *cols)
{
   FILE *fp;
   char buf[71];
   int p, size;

   /***************************************************************************
   * Open the input image file for reading if a filename was given. If no
   * filename was provided, set fp to read from standard input.
   ***************************************************************************/
   if(infilename == NULL) fp = stdin;
   else{
      if((fp = fopen(infilename, "r")) == NULL){
         fprintf(stderr, "Error reading the file %s in read_ppm_image().\n",
            infilename);
         return(0);
      }
   }

   /***************************************************************************
   * Verify that the image is in PPM format, read in the number of columns
   * and rows in the image and scan past all of the header information.
   ***************************************************************************/
   fgets(buf, 70, fp);
   if(strncmp(buf,"P6",2) != 0){
      fprintf(stderr, "The file %s is not in PPM format in ", infilename);
      fprintf(stderr, "read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
   sscanf(buf, "%d %d", cols, rows);
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

   /***************************************************************************
   * Allocate memory to store the image then read the image from the file.
   ***************************************************************************/
   if(((*image_red) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if(((*image_grn) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if(((*image_blu) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }

   size = (*rows)*(*cols);
   for(p=0;p<size;p++){
      (*image_red)[p] = (unsigned char)fgetc(fp);
      (*image_grn)[p] = (unsigned char)fgetc(fp);
      (*image_blu)[p] = (unsigned char)fgetc(fp);
   }

   if(fp != stdin) fclose(fp);
   return(1);
}

/******************************************************************************
* Function: write_ppm_image
* Purpose: This function writes an image in PPM format. The file is either
* written to the file specified by outfilename or to standard output if
* outfilename = NULL. A comment can be written to the header if coment != NULL.
******************************************************************************/
int write_ppm_image(char *outfilename, unsigned char *image_red,
    unsigned char *image_grn, unsigned char *image_blu, int rows,
    int cols, char *comment, int maxval)
{
   FILE *fp;
   long size, p;

   /***************************************************************************
   * Open the output image file for writing if a filename was given. If no
   * filename was provided, set fp to write to standard output.
   ***************************************************************************/
   if(outfilename == NULL) fp = stdout;
   else{
      if((fp = fopen(outfilename, "w")) == NULL){
         fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
            outfilename);
         return(0);
      }
   }

   /***************************************************************************
   * Write the header information to the PGM file.
   ***************************************************************************/
   fprintf(fp, "P6\n%d %d\n", cols, rows);
   if(comment != NULL)
      if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);

   /***************************************************************************
   * Write the image data to the file.
   ***************************************************************************/
   size = (long)rows * (long)cols;
   for(p=0;p<size;p++){      /* Write the image in pixel interleaved format. */
      fputc(image_red[p], fp);
      fputc(image_grn[p], fp);
      fputc(image_blu[p], fp);
   }

   if(fp != stdout) fclose(fp);
   return(1);
}
