#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "hist-equ.h"
#include "CudaErrorChecks.h"




#ifdef USE_CONSTANT
__constant__ int const_lut[256];


//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 67ms [1x1024])			DEACLARED HERE BECAUSE OF CONSTANT MEMORY
__global__ void histogram_equalizationGPU4(unsigned char *img_out, unsigned char *img_in,
                                  int img_size)
{	
	int ix = threadIdx.x;

	for (int i = ix; i < img_size; i += 1024 )
		img_out[i] = (unsigned char)const_lut[img_in[i]];

}

#endif


int check_histogram (int *CPUhist, int *GPUhist)
{
	int error = 0;
	for (int i = 0; i < 256; i++){
		error += GPUhist[i] - CPUhist[i];
		if(error !=0)
			break;
	}
	
	if (error == 0)
		return (0);
	else
		return (1);
}

int check_results (unsigned char *CPU_result, unsigned char *GPU_result, int img_size)
{
	/*
	int sum = 0;
	for (int i = 0; i < img_size; i++)
		sum += CPU_result[i];
	printf("\n\tHost Image Sum   = %2d\n", sum);
	
	sum = 0;
	for (int i = 0; i < img_size; i++)
		sum += GPU_result[i];
	printf("\tDevice Image Sum = %2d\n", sum);
*/

	int error = 0;
	for (int i = 0; i < img_size; i++){
		error += GPU_result[i] - CPU_result[i];
		if(error !=0)
			break;
	}	

	if (error == 0)
		return (0);
	else
		return (1);
}


PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
	////////////////////////////////////////////////////////////////////////////
	int width    = img_in.w;
	int height   = img_in.h;
	int img_size = width * height;
	int img_size_B = img_size * sizeof(unsigned char);
	int hist_size_B = (sizeof(int)) << 8;
	int lut_size_B = (sizeof(int)) *256;
	////////////////////////////////////////////////////////////////////////////	
	


	/*Timing variables*/
	struct timespec  tv1, tv2;
	double h_histo_time, h_eq_time, h_total_time;
	cudaEvent_t start, stop;
	float GPU_time = 0.0,
		  GPU_time_transfer = 0.0,
		  GPU_time_calc = 0.0,
		  GPU_time_transfer_to_d_Hist = 0.0,
		  GPU_time_transfer_to_d_Eq = 0.0,
		  GPU_time_transfer_from_d_Hist = 0.0,
		  GPU_time_transfer_from_d_Eq = 0.0,
		  GPU_time_calc_hist = 0.0,
		  GPU_time_calc_eq = 0.0,
		  create_lut_time = 0.0;

    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(img_size_B);
    
	printf(ANSI_COLOR_YELLOW"\nStarting CPU processing...\n"ANSI_COLOR_WHITE);

#ifdef TIMING
double cpuTries[12], gpuTries[12],
			cpuTime[10], gpuTime[10];
double average, std_deviation, sum = 0, sum1 = 0,swap, variance;
int c, d;

for(int j = 0; j<12; j++){
#endif

	/*Get the starting time.*/	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);

	/*Take the end time	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
	h_histo_time =  (double)(
					(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
					(double) (tv2.tv_sec - tv1.tv_sec)  ) * 1000.0;



	/*Get the starting time.*/	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

	/*Take the end time	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	h_eq_time = (double)(
				(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
				(double) (tv2.tv_sec - tv1.tv_sec)  ) * 1000.0;

	h_total_time = h_histo_time + h_eq_time;

#ifdef TIMING
cpuTries[j]= h_total_time;

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
	for(int i=1; i<11;i++){
		cpuTime[i-1] = cpuTries[i];
	}

	/*  Compute the sum of all elements */
	for (int i = 0; i < 10; i++){
		sum = sum + cpuTime[i];
	}

	average = sum / (double)10;
	/*  Compute  variance  and standard deviation  */
	for (int i = 0; i < 10; i++){
		sum1 = sum1 + pow((cpuTime[i] - average), 2);
	}
	variance = sum1 / (float)10;
	std_deviation = sqrt(variance);
	//printf("Average in CPU = %.7lf\n", average);
	//printf("variance of all elements = %.5lf\n", variance);
	//printf("Standard deviation in CPU = %.7lf\n", std_deviation);
#endif


	printf(ANSI_COLOR_YELLOW"...CPU processing finished\n"ANSI_COLOR_WHITE);

	printf(ANSI_COLOR_YELLOW"\n==============TIMES=============="ANSI_COLOR_WHITE);
	printf (ANSI_COLOR_YELLOW"\nCPU histogram time: %.7g ms\n"ANSI_COLOR_WHITE,h_histo_time);
	printf (ANSI_COLOR_YELLOW"CPU equalization time: %.7g ms\n"ANSI_COLOR_WHITE,h_eq_time);
	printf (ANSI_COLOR_YELLOW"CPU total time: %.7g ms\n"ANSI_COLOR_WHITE,h_total_time);
	
#ifdef TIMING
	printf(ANSI_COLOR_YELLOW"\n==============SERIAL CODE STATISTICS==============\n");
	printf("Average in CPU = %.7lf\n", average);
	printf("Standard deviation in CPU = %.7lf\n"ANSI_COLOR_WHITE, std_deviation);
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////           Calling GPU KERNELS                //////////////////
////////////////////////////////////////////////////////////////////////////////	

	// Declare and initialize histogram used for the result of GPU computation
	int *h_hist = 0;
	h_hist = (int *)malloc(hist_size_B);
	//memset(h_hist, 0, 256 * sizeof(int));

	//Declare device variables
	int *d_hist = 0;			//device histogram
	unsigned char *d_img = 0;	//device image

	/*create gpu timers*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Allocate memory for device */
	printf(ANSI_COLOR_GREEN"\nAllocating and initializing device arrays...\n"ANSI_COLOR_WHITE);
	cudaSetDevice(0);
	CudaSafeCall( cudaMalloc((void **)&d_hist, hist_size_B) );
	CudaSafeCall( cudaMalloc((void **)&d_img, img_size_B) );


	/* Initialize device memory */
	CudaSafeCall( cudaMemset(d_hist,	0, hist_size_B) );
	CudaSafeCall( cudaMemset(d_img,		0, img_size_B) );

	printf(ANSI_COLOR_GREEN"Allocating device arrays finished...\n\n"ANSI_COLOR_WHITE);

	
	

	cudaError_t ce = cudaSuccess; 
	int device; 
    cudaDeviceProp  prop; 
    int numBlocks; 
 

    ce = cudaGetDevice(&device); 
    ce = cudaGetDeviceProperties( &prop, device ); 
    numBlocks = prop.multiProcessorCount; 
	printf("Using device: %d , with: %d multiProcessors \n", device,numBlocks);
 
    printf(ANSI_COLOR_GREEN"\nStarting GPU processing...\n"ANSI_COLOR_WHITE);

	/*kernel geometry*/
	dim3 block_size;
	block_size.x = 256;

	/* n x n grid*/
	dim3 grid_size;
	grid_size.x = numBlocks * 2;

	PGM_IMG h_result;
	h_result.w = img_in.w;
	h_result.h = img_in.h;

	unsigned char *d_result = 0;
	int *d_lut = 0;
	CudaSafeCall( cudaMalloc((void **)&d_result, img_size_B) );
	CudaSafeCall( cudaMalloc((void **)&d_lut, lut_size_B) );

	/* Initialize device memory */
	CudaSafeCall( cudaMemset(d_result,	0, img_size_B) );
	CudaSafeCall( cudaMemset(d_lut,		0, lut_size_B) );

	int *h_lut =0;


#ifdef TIMING
for(int j = 0; j<12; j++){
sum = 0;
sum1 = 0;
#endif
memset(h_hist, 0, 256 * sizeof(int));
	
	cudaEventRecord(start);
	CudaSafeCall( cudaMemcpy(d_hist, h_hist, hist_size_B, 	cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_img, img_in.img, img_size_B, cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_transfer_to_d_Hist, start, stop);

	

	///////212, 256 gia method_1.....(1024,256) gia method_2
	cudaEventRecord(start);
	histogramGPU_2<<<1024,256>>>(d_hist, d_img, img_size); //Histogram kernel
	CudaCheckError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_hist, start, stop);

	//Copy results back to host
	cudaEventRecord(start);
	CudaSafeCall( cudaMemcpy(h_hist, d_hist, hist_size_B, 	cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_transfer_from_d_Hist, start, stop);


	//Calculate lut and pass it to GPU
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	create_lut(hist, h_result.w * h_result.h, 256, &h_lut);

	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
	create_lut_time =	(double)(
						(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
						(double) (tv2.tv_sec - tv1.tv_sec)  ) * 1000.0;


	//move data to device
	cudaEventRecord(start);
#ifdef USE_CONSTANT
	CudaSafeCall( cudaMemcpyToSymbol(const_lut, h_lut, lut_size_B) );		//Constant memory
#else
	CudaSafeCall( cudaMemcpy(d_lut, h_lut, lut_size_B, cudaMemcpyHostToDevice) );
#endif
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_transfer_to_d_Eq, start, stop);
	
// Choose method

/*naive - WORKS FINE*/
/*
	cudaEventRecord(start);
	histogram_equalizationGPU1<<<1, 1024>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/


/*naive2 - WORKS FINE*/
/*
	cudaEventRecord(start);
	histogram_equalizationGPU2<<<8, 256>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/


/*Shared memory - WORKS FINE*/
/*
	cudaEventRecord(start);
	histogram_equalizationGPU3<<<1, 1024>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Constant memory - WORKS FINE*/
/*
	cudaEventRecord(start);
	histogram_equalizationGPU4<<<1, 1024>>>(d_result, d_img, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/


/*Shared memory - WORKS FINE*/
/*
	block_size.x = 1024;
	grid_size.x = img_size/1024;
	if ( img_size % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU5<<<grid_size, block_size>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Shared memory - WORKS FINE.*/
/*
	block_size.x = 1024;
	grid_size.x = (img_size/4)/1024;
	if ( (img_size/4) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU6<<<grid_size, block_size>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Shared memory - WORKS FINE. */
/*
	block_size.x = 1024;
	grid_size.x = (img_size/8)/1024;
	if ( (img_size/8) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU7<<<grid_size, block_size>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Shared memory - WORKS FINE. BEST*/

	block_size.x = 1024;
	grid_size.x = (img_size/16)/1024;
	if ( (img_size/16) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU8<<<grid_size, block_size>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);


/*Shared memory-passing size as parameter to kernel - WORKS FINE. */
/*
	block_size.x = 256;
	grid_size.x = (img_size/8)/block_size.x;
	if ( (img_size/8) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU9<<<grid_size, block_size, (8*block_size.x*sizeof(unsigned char))>>>(d_result, d_img, d_lut, img_size, block_size.x);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Shared memory - WORKS FINE. */
/*
	block_size.x = 1024;
	grid_size.x = (img_size/32)/1024;
	if ( (img_size/32) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU10<<<grid_size, block_size>>>(d_result, d_img, d_lut, img_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

/*Shared memory-passing size as parameter to kernel - WORKS FINE.*/
/*
	block_size.x = 256;
	grid_size.x = (img_size/16)/block_size.x;
	if ( (img_size/16) % block_size.x)
		grid_size.x += 1;
	cudaEventRecord(start);
	histogram_equalizationGPU11<<<grid_size, block_size, (16*block_size.x*sizeof(unsigned char))>>>(d_result, d_img, d_lut, img_size, block_size.x);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_calc_eq, start, stop);
*/

	//Copy results back to host
	h_result.img = (unsigned char *)malloc(img_size_B);
	cudaEventRecord(start);
	CudaSafeCall( cudaMemcpy(h_result.img, d_result, img_size_B, cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_time_transfer_from_d_Eq, start, stop);

	//Calculate Times
	GPU_time_transfer = GPU_time_transfer_to_d_Hist + GPU_time_transfer_from_d_Hist
						+ GPU_time_transfer_to_d_Eq + GPU_time_transfer_from_d_Eq;

	GPU_time_calc = GPU_time_calc_hist + create_lut_time + GPU_time_calc_eq;

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
	for(int i=1; i<11;i++){
		gpuTime[i-1] = gpuTries[i];
	}

	/*  Compute the sum of all elements */
	for (int i = 0; i < 10; i++){
		sum = sum + gpuTime[i];
	}

	average = sum / (double)10;
	/*  Compute  variance  and standard deviation  */
	for (int i = 0; i < 10; i++){
		sum1 = sum1 + pow((gpuTime[i] - average), 2);
	}
	variance = sum1 / (float)10;
	std_deviation = sqrt(variance);
	//printf("Average in GPU = %.7lf\n", average);
	//printf("variance of all elements = %.5lf\n", variance);
	//printf("Standard deviation in GPU = %.7lf\n", std_deviation);
#endif



	printf(ANSI_COLOR_GREEN"\n...GPU processing finished\n\n"ANSI_COLOR_WHITE);


	/*Error Checking*/
	//Check Histograms
	if ( check_histogram(hist, h_hist) == 0)
		printf("\t\tHistograms are indetical!\n");
	else
		printf("\e[4;31m""\t\t~~~~~Histograms are different!~~~~~\n"ANSI_COLOR_WHITE);

	//Check Images
	if ( check_results(result.img, h_result.img, img_size) == 0)
		printf("\t\tImages are indetical!\n");
	else
		printf("\e[4;31m""\t\t~~~~~Images are different!~~~~~\n"ANSI_COLOR_WHITE);


	


 	printf(ANSI_COLOR_GREEN"\n==============TIMES==============");
	printf ("\nGPU time for hist %.7g ms\n", GPU_time_transfer_to_d_Hist + GPU_time_transfer_from_d_Hist + GPU_time_calc_hist );
	printf ("GPU time for eq %.7g ms\n", GPU_time_transfer_to_d_Eq + GPU_time_transfer_from_d_Eq + create_lut_time + GPU_time_calc_eq);
	printf ("Total GPU and CPU time %.7g ms\n"ANSI_COLOR_WHITE, GPU_time);

#ifdef TIMING
	printf(ANSI_COLOR_GREEN"\n============== PARALLEL/OPTIMIZED CODE STATISTICS==============\n");
	printf("Average in optimized = %.7lf\n", average);
	printf("Standard deviation in optimized = %.7lf\n"ANSI_COLOR_WHITE, std_deviation);
#endif

	printf(ANSI_COLOR_RED"\n==============BREAKPOINTS==============\n");
	printf ("\tGPU_time_transfer_to_d_Hist %.7g ms\n", GPU_time_transfer_to_d_Hist);
	printf ("\tGPU_time_transfer_from_d_Hist %.7g ms\n", GPU_time_transfer_from_d_Hist);
	printf ("\tGPU_time_calc_hist %.7g ms\n", GPU_time_calc_hist);
	printf ("\tGPU_time_transfer_to_d_Eq %.7g ms\n", GPU_time_transfer_to_d_Eq);
	printf ("\tGPU_time_transfer_from_d_Eq %.7g ms\n", GPU_time_transfer_from_d_Eq);
	printf ("\tcreate_lut_time %.7g ms\n", create_lut_time);
	printf ("\tGPU_time_calc_eq %.7g ms\n"ANSI_COLOR_WHITE, GPU_time_calc_eq);

	




#ifdef CUDA_ERROR_CHECK
	cudaThreadSynchronize();									//remove when code has no bugs
#endif



	// ask CUDA for the last error to occur (if one exists)
/*
	cudaError_t error = cudaGetLastError();						
	if(error != cudaSuccess){
		// something's gone wrong
		// print out the CUDA error as a string
		printf("CUDA Error: %s\n", cudaGetErrorString(error));

		// we can't recover from the error -- exit the program
		return 1;
	}
	// no error occurred, proceed as usual
*/

	// free all the allocated memory
	free(h_hist);
	free(h_lut);

	CudaSafeCall(cudaFree(d_hist));
	CudaSafeCall(cudaFree(d_img));
	CudaSafeCall(cudaFree(d_result));
	CudaSafeCall(cudaFree(d_lut));

	// Do a device reset just in case... 
    cudaDeviceReset();


    return result;
}



























