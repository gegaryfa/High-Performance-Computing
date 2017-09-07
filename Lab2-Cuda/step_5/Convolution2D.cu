/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

unsigned int filter_radius;

#define ANSI_COLOR_WHITE   "\x1B[0m"	//WHITE
#define ANSI_COLOR_GREEN   "\x1b[32m"   //GREEN color for GPU messages :)
#define ANSI_COLOR_RED     "\x1b[31m"	//RED


// Define this to turn on error checking
#define CUDA_ERROR_CHECK
 
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.005
#define is_power_of_2(num)	(0 == ((num != 1) && (num & (num - 1))))


/*
 *Switch between double and single precision
 *Default: float
 *Use -USE_DOUBLE=1 as a flag at compilation to use double precision
 */


//#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif


//#define TIMING


void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}
 
void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}





////////////////////////////////////////////////////////////////////////////////
// CPU Functions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(real *h_Dst, real *h_Src, real *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {	//col
    for (x = 0; x < imageW; x++) {	//row
      real sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

     
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(real *h_Dst, real *h_Src, real *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      real sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
// GPU Kernels
////////////////////////////////////////////////////////////////////////////////

/*__global__ void convolutionRowGPU(real *d_Dst,	(out)-->rowConvoled image
 *							real *d_Src, 			(in) -->Image to be convoled
 *							real *d_Filter, 		(in) -->Filter for convolution
 *							int filterR) 			(in) -->Filter Radius
 *
 *
*/
__global__ void convolutionRowGPU(real *d_Dst,
 							real *d_Src, 
							real *d_Filter, 
							int filterR) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;	//col
	int iy = blockIdx.y * blockDim.y + threadIdx.y;	//row
	
	int grid_width = gridDim.x * blockDim.x; 
	int idx = iy * grid_width + ix;					//destination

	real sum = 0;
	int k;

	for (k = -filterR; k <= filterR; k++) {
		int d = ix + k;

		if (d >= 0 && d < grid_width) {
			sum += d_Src[iy * grid_width + d] * d_Filter[filterR - k];
		}     

		d_Dst[idx] = sum;
	}
      
}


/*__global__ void convolutionColumnGPU(real *d_Dst,	(out)-->rowConvoled image
 *							real *d_Src, 			(in) -->Image to be convoled
 *							real *d_Filter, 		(in) -->Filter for convolution
 *							int filterR) 			(in) -->Filter Radius
 *
 *
*/
__global__ void convolutionColumnGPU(real *d_Dst,
 							real *d_Src, 
							real *d_Filter, 
							int filterR) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;	//col
	int iy = blockIdx.y * blockDim.y + threadIdx.y;	//row
	
	int grid_width = gridDim.x * blockDim.x; 
	int idx = iy * grid_width + ix;					//destination

	real sum = 0;
	int k;

	for (k = -filterR; k <= filterR; k++) {
		int d = iy + k;

		if (d >= 0 && d < grid_width) {
			sum += d_Src[d * grid_width + ix] * d_Filter[filterR - k];
		}     

		d_Dst[idx] = sum;
	}	

}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
	/*Host pointers (CPU)*/
    real
    *h_Filter = 0,
    *h_Input = 0,
    *h_Buffer = 0,
    *h_OutputCPU = 0,
	*h_OutputGPU = 0;

	/*Device pointers (GPU)*/
    real
    *d_Filter = 0,
    *d_Input = 0,
    *d_Buffer = 0,
    *d_OutputGPU = 0;


    int imageW;
    int imageH;
	int TILE_W = 32;			//TILE_W x TILE_W blocks inside the grid	
    unsigned int i, j;

	struct timespec  tv1, tv2;
	cudaEvent_t start, stop;
	float GPU_time = 0.0,
		   GPU_time_transfer = 0.0,
		   GPU_time_calc = 0.0;


	do{
		printf("Enter filter radius : ");
		scanf("%d", &filter_radius);
	}while(filter_radius <=0);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

	do{
		printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
		scanf("%d", &imageW);
	}while(imageW <= FILTER_LENGTH || !is_power_of_2(imageW));

    imageH = imageW;

	unsigned int sizeof_image = imageW * imageH * sizeof(real);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

	/* Allocate memory for host */
    h_Filter    = (real *)malloc(FILTER_LENGTH * sizeof(real));
    h_Input     = (real *)malloc(sizeof_image);
    h_Buffer    = (real *)malloc(sizeof_image);
    h_OutputCPU = (real *)malloc(sizeof_image);
	h_OutputGPU = (real *)malloc(sizeof_image);

	// if memory allocation on host failed, report an error message
	if(h_Filter == 0 || h_Input == 0 || h_Buffer ==0 || h_OutputCPU == 0 || h_OutputGPU == 0 ){
		printf("couldn't allocate memory\n");
		return 1;
	}
	printf("Allocating host arrays finished...\n\n");

	/* Allocate memory for device */
	printf("Allocating and initializing device arrays...\n");
	cudaSetDevice(0);
	CudaSafeCall( cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(real)) );
	CudaSafeCall( cudaMalloc((void **)&d_Input,  sizeof_image) );
	CudaSafeCall( cudaMalloc((void **)&d_Buffer, sizeof_image) );
	CudaSafeCall( cudaMalloc((void **)&d_OutputGPU, sizeof_image) );

	/* Initialize device memory */
	CudaSafeCall( cudaMemset(d_Filter,    0.f, FILTER_LENGTH * sizeof(real)) );
	CudaSafeCall( cudaMemset(d_Input,     0.f, sizeof_image) );
	CudaSafeCall( cudaMemset(d_Buffer,    0.f, sizeof_image) );
	CudaSafeCall( cudaMemset(d_OutputGPU, 0.f, sizeof_image) );
	
	printf("Allocating device arrays finished...\n\n");

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (real)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (real)rand() / ((real)RAND_MAX / 255) + (real)rand() / (real)RAND_MAX;
    }


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

#ifdef TIMING
double cpuTries[12], gpuTries[12],
			cpuTime[10], gpuTime[10];
double average, std_deviation, sum = 0, sum1 = 0,swap, variance;
int c, d;

for(j = 0; j<12; j++){

	/*Get the starting time.*/	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

	/*Take the end time	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	cpuTries[j]=((double)(
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec)  ) * 1000.0);

}

/*sort*/
	for (c = 0 ; c < ( 11 ); c++){
		for (d = 0 ; d < 12 - c - 1; d++){
			if (cpuTries[d] > cpuTries[d+1]) /* For decreasing order use < */
			{
		    swap = cpuTries[d];
		    cpuTries[d] = cpuTries[d+1];
		    cpuTries[d+1] = swap;
		  }
		}
  	}	
	/*through out max and min*/ /*LATHOS*/
	for(i=1; i<11;i++){
		cpuTime[i-1] = cpuTries[i];
	}

	/*  Compute the sum of all elements */
	for (i = 0; i < 10; i++){
		sum = sum + cpuTime[i];
	}

	average = sum / (double)10;
	/*  Compute  variance  and standard deviation  */
	for (i = 0; i < 10; i++){
		sum1 = sum1 + pow((cpuTime[i] - average), 2);
	}
	variance = sum1 / (float)10;
	std_deviation = sqrt(variance);
	printf("Average in CPU = %.7lf\n", average);
	//printf("variance of all elements = %.5lf\n", variance);
	printf("Standard deviation in CPU = %.7lf\n", std_deviation);

#else
	/*Get the starting time.*/	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

	/*Take the end time	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
#endif


	printf("CPU computation finished...\n\n");



	/*create gpu timers*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	printf(ANSI_COLOR_GREEN	"GPU computation..." ANSI_COLOR_WHITE "\n");

#ifdef TIMING
for(j = 0; j<12; j++){
sum = 0;
sum1 = 0;
#endif

	/* Copy filter and image to device memory  */
		
	cudaEventRecord(start);
	CudaSafeCall( cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(real), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_Input, h_Input, sizeof_image, cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&GPU_time_transfer, start, stop);

	/* imageH x imageW threads per block */
	dim3 block_size;
	block_size.x = TILE_W;
	block_size.y = TILE_W;

	/* n x n grid*/
	dim3 grid_size;
	grid_size.x = imageW/block_size.x;
	grid_size.y = imageH/block_size.y;

	cudaEventRecord(start);
	convolutionRowGPU<<<grid_size,block_size>>>(d_Buffer, d_Input, d_Filter, filter_radius); // convolution kata grammes
	CudaCheckError();
    convolutionColumnGPU<<<grid_size,block_size>>>(d_OutputGPU, d_Buffer, d_Filter, filter_radius); // convolution kata sthles
	CudaCheckError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&GPU_time_calc, start, stop);

	/* Copy the result back to host*/
	cudaEventRecord(start);
	CudaSafeCall( cudaMemcpy(h_OutputGPU, d_OutputGPU, sizeof_image, cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	GPU_time_transfer += cudaEventElapsedTime(&GPU_time_transfer, start, stop);

	GPU_time = GPU_time_transfer + GPU_time_calc;
	
#ifdef TIMING
	gpuTries[j] = GPU_time;
}

/*sort*/
	for (c = 0 ; c < ( 11 ); c++){
		for (d = 0 ; d < 12 - c - 1; d++){
			if (gpuTries[d] > gpuTries[d+1]) /* For decreasing order use < */
			{
		    swap = gpuTries[d];
		    gpuTries[d] = gpuTries[d+1];
		    gpuTries[d+1] = swap;
		  }
		}
  	}	
	/*through out max and min*/ /*LATHOS*/
	for(i=1; i<11;i++){
		gpuTime[i-1] = gpuTries[i];
	}

	/*  Compute the sum of all elements */
	for (i = 0; i < 10; i++){
		sum = sum + gpuTime[i];
	}

	average = sum / (double)10;
	/*  Compute  variance  and standard deviation  */
	for (i = 0; i < 10; i++){
		sum1 = sum1 + pow((gpuTime[i] - average), 2);
	}
	variance = sum1 / (float)10;
	std_deviation = sqrt(variance);
	printf("Average in GPU = %.7lf\n", average);
	//printf("variance of all elements = %.5lf\n", variance);
	printf("Standard deviation in GPU = %.7lf\n", std_deviation);
#endif


	printf(ANSI_COLOR_GREEN	"GPU computation finished..." ANSI_COLOR_WHITE "\n\n");

	bool error_check = false;
	real errorValue = 0;
	//real max_error = 0;
	for(j = 0; j<imageW * imageH; j++){
		if(ABS(h_OutputGPU[j] - h_OutputCPU[j]) > accuracy){
			//max_error = ABS(h_OutputGPU[j] - h_OutputCPU[j]);
			errorValue = ABS(h_OutputGPU[j] - h_OutputCPU[j]);
			error_check = true;			
			break;
		}
	}
	

	printf("\n=============RESULTS=============\n");

	if(error_check == true){
		printf(ANSI_COLOR_RED	"The images differ \n" ANSI_COLOR_WHITE);
		printf(ANSI_COLOR_RED   "for :%f \n" ANSI_COLOR_WHITE, errorValue );	
	}
	else
		printf(ANSI_COLOR_GREEN	"The images are identical \n" ANSI_COLOR_WHITE );


	

	printf("\n==============TIMES==============");

	printf ("\nCPU time %.7g ms\n",(double)(
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec)  ) * 1000.0);					//* 1000.0 gia ms

	printf ("\nGPU time %.7g ms\n", GPU_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// make the host wait until the kernel is finished executing before
	// checking for the last CUDA error. otherwise, we won't detect
	// an error if one occurred
#ifdef CUDA_ERROR_CHECK
	cudaThreadSynchronize();									//remove when code has no bugs
#endif
	// this kind of "blocking" operation is usually only appropriate during
	// the debugging process because it forces the host to wait on the device
	// while it could be busy doing other things. once the code has been
	// debugged, frequent error checking code should be eliminated or disabled

	// ask CUDA for the last error to occur (if one exists)

	cudaError_t error = cudaGetLastError();						
	if(error != cudaSuccess){
		// something's gone wrong
		// print out the CUDA error as a string
		printf("CUDA Error: %s\n", cudaGetErrorString(error));

		// we can't recover from the error -- exit the program
		return 1;
	}
	// no error occurred, proceed as usual

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  



    // free all the allocated memory
    free(h_OutputGPU);
	free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);

	CudaSafeCall(cudaFree(d_OutputGPU));
	CudaSafeCall(cudaFree(d_Buffer));
	CudaSafeCall(cudaFree(d_Input));
	CudaSafeCall(cudaFree(d_Filter));

    // Do a device reset just in case... 
     cudaDeviceReset();


    return 0;
}
