// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

#define TIMING

/* The horizontal and vertical operators to be used in the sobel filter */
char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
//int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */

/*This function was moved into sobel function*/
/*
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]) {
	int i, j, res;
  
	res = 0;
	*/
	/*Loop Unroll No1*/
	/*i=-1*/
	/*
	res += input[(posy - 1)*SIZE + posx - 1] * operator[-1+1][-1+1];
	res += input[(posy - 1)*SIZE + posx + 0] * operator[-1+1][0+1];
	res += input[(posy - 1)*SIZE + posx + 1] * operator[-1+1][1+1];
	*/
	/*i=0*/
	/*
	res += input[(posy)*SIZE + posx - 1] * operator[0+1][-1+1];
	res += input[(posy)*SIZE + posx + 0] * operator[0+1][0+1];
	res += input[(posy)*SIZE + posx + 1] * operator[0+1][1+1];
	*/
	/*i=1*/
	/*
	res += input[(posy + 1)*SIZE + posx - 1] * operator[1+1][-1+1];
	res += input[(posy + 1)*SIZE + posx + 0] * operator[1+1][0+1];
	res += input[(posy + 1)*SIZE + posx + 1] * operator[1+1][1+1];
	*/
	/*	
	for (i = -1; i <= 1; i++) {
		for (j = -1; j <= 1; j++) {
			res += input[(posy + i)*SIZE + posx + j] * operator[i+1][j+1];
		}
	}
	*/
	/*return(res);
}
*/

/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int i, j, res1, res2;
	unsigned int p;
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	/*Loop Interchange No2*/
	for (i=1; i<SIZE-1; i+=1 ) {
		for (j=1; j<SIZE-1; j+=1) {		
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.
								  */
			/*Function inlining (convolution2D --> sobel)*/
			/*
			 *p = pow(convolution2D(i, j, input, horiz_operator), 2) + * 
			 *	pow(convolution2D(i, j, input, vert_operator), 2);	   *
			*/
			
			res1=0;
			/*i=-1*/
			res1 += input[(i - 1)*SIZE + j - 1] * horiz_operator[-1+1][-1+1];
			res1 += input[(i - 1)*SIZE + j + 0] * horiz_operator[-1+1][0+1];
			res1 += input[(i - 1)*SIZE + j + 1] * horiz_operator[-1+1][1+1];
			/*i=0*/
			res1 += input[(i)*SIZE + j - 1] * horiz_operator[0+1][-1+1];
			res1 += input[(i)*SIZE + j + 0] * horiz_operator[0+1][0+1];
			res1 += input[(i)*SIZE + j + 1] * horiz_operator[0+1][1+1];
			/*i=1*/
			res1 += input[(i + 1)*SIZE + j - 1] * horiz_operator[1+1][-1+1];
			res1 += input[(i + 1)*SIZE + j + 0] * horiz_operator[1+1][0+1];
			res1 += input[(i + 1)*SIZE + j + 1] * horiz_operator[1+1][1+1];
	
			res2=0;
			/*i=-1*/
			res2 += input[(i - 1)*SIZE + j - 1] * vert_operator[-1+1][-1+1];
			res2 += input[(i - 1)*SIZE + j + 0] * vert_operator[-1+1][0+1];
			res2 += input[(i - 1)*SIZE + j + 1] * vert_operator[-1+1][1+1];
			/*i=0*/
			res2 += input[(i)*SIZE + j - 1] * vert_operator[0+1][-1+1];
			res2 += input[(i)*SIZE + j + 0] * vert_operator[0+1][0+1];
			res2 += input[(i)*SIZE + j + 1] * vert_operator[0+1][1+1];
			/*i=1*/
			res2 += input[(i + 1)*SIZE + j - 1] * vert_operator[1+1][-1+1];
			res2 += input[(i + 1)*SIZE + j + 0] * vert_operator[1+1][0+1];
			res2 += input[(i + 1)*SIZE + j + 1] * vert_operator[1+1][1+1];

			p = pow(res1,2) + pow(res2,2);

			res = (int)sqrt(p);

			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			if (res > 255)
				output[i*SIZE + j] = 255;      
			else
				output[i*SIZE + j] = (unsigned char)res;
		}
	}

	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */
	/*Loop Interchange No3 -->leads to worst results*/	
	for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-1; j++ ) {
			t = pow((output[i*SIZE+j] - golden[i*SIZE+j]),2);
			PSNR += t;
		}
	}
  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	/*printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
	*/

	/*results without text*/
	printf ("%.7g\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;

	#ifdef TIMING
	int j;
	for (j = 0; j < 12; ++j){
		PSNR = sobel(input, output, golden);
		//sleep(2);
	}
	#endif

	#ifdef SHOW
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");
	#endif

	return 0;
}

