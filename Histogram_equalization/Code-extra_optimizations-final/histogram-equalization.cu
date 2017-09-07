#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "hist-equ.h"
#include "CudaErrorChecks.h"

////////////////////////////////////////////////////////////////////////////////
////////////////		CPU KERNELS				////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void histogram (int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization (unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
    }

    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}


void  create_lut(int * hist_in, int img_size, int nbr_bin, int **lut)
{
    *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        (*lut)[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if((*lut)[i] < 0){
            (*lut)[i] = 0;
        }   
    }
	

}




////////////////////////////////////////////////////////////////////////////////
////////////////		GPU KERNELS				////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void histogramGPU_1 (int *hist_out, unsigned char *img_in, int img_size)
{
	__shared__ int cache[256];
	cache[threadIdx.x] = 0;
	__syncthreads();
	
	int id = blockIdx.x * blockDim.x + threadIdx.x ;
	const int jump = gridDim.x * blockDim.x;
	
	while (id < img_size) {
		atomicAdd(&cache[img_in[id]], 1);
		id += jump;
	}
	__syncthreads();

	atomicAdd(&hist_out[threadIdx.x], cache[threadIdx.x]);
}



__global__  void histogramGPU_2 (int* hist_out, unsigned char *img_in, int img_size) 
{ 
    int ix = blockIdx.x * blockDim.x + threadIdx.x; 	//column se grammiko filtro
 
    if (ix > img_size) 
        return; 
 
    // Create shared buffer size of threads per block and clear it 
    // Size of array = #numBins 
    __shared__ int cache[256]; 
    cache[threadIdx.x] = 0; 
    __syncthreads(); 
  
    int jump = blockDim.x * gridDim.x;  
 
    // Each thread walks through several data
	#pragma unroll
    while (ix < img_size) 
    { 
        int idx = img_in[ix]; 
        atomicAdd(&(cache[idx]), 1); 
        ix += jump; 
    } 
 
    __syncthreads(); 
 
    // Update global memory 
    atomicAdd(&(hist_out[threadIdx.x]), cache[threadIdx.x]); 
}




	//just for the result image		(WORKS FINE. TIME FOR 6400x6400 image =~ 77ms)
__global__ void histogram_equalizationGPU1(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	
	int ix = threadIdx.x;
	int i;

	for (i = ix; i < img_size; i += 1024)
		img_out[i] = (unsigned char)lut[img_in[i]];
}




	//just for the result image		(WORKS FINE. TIME FOR 6400x6400 image =~ 0.79ms [15x256])
__global__ void histogram_equalizationGPU2(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	int ix = blockIdx.x * blockDim.x + threadIdx.x; 	//column se grammiko filtro
 
    if (ix > img_size) 
        return; 

	int i;

	for (i = ix; i < img_size; i += 256)
		img_out[i] = (unsigned char)lut[img_in[i]];
		

}


	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 50ms [1x1024])
__global__ void histogram_equalizationGPU3(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	__shared__ int cache_lut[256];

	int ix = threadIdx.x;
	if(ix <256)
		cache_lut[ix] = lut[ix];
	__syncthreads();


	for (int i = ix; i < img_size; i += 1024)
		img_out[i] = (unsigned char)cache_lut[img_in[i]];

}


	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 35ms [1x1024])
__global__ void histogram_equalizationGPU5(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	int ix = threadIdx.x;	//col
	int idx = blockIdx.x*blockDim.x + ix;			//destination

	if(idx>img_size)
		return;

	__shared__ int cache_lut[256];
	__shared__ unsigned char cache_img_buffer[1024];
							

	if(threadIdx.x <256)
		cache_lut[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();								


	cache_img_buffer[threadIdx.x] = (unsigned char)cache_lut[img_in[idx]];
	//__syncthreads();
	img_out[idx] = cache_img_buffer[threadIdx.x];
}


	
	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 34ms [1x1024])		//4 times shared memory and work/thread
__global__ void histogram_equalizationGPU6(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	int ix = threadIdx.x;	//col
	int idx = blockIdx.x*blockDim.x + ix;			//destination

	if(idx > img_size) 
		return;

	__shared__ int cache_lut[256];
	__shared__ unsigned char cache_img_buffer[4096];
							

	if(threadIdx.x <256)
		cache_lut[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();							


	cache_img_buffer[threadIdx.x] = (unsigned char)cache_lut[img_in[idx]];
	cache_img_buffer[threadIdx.x+1024] = (unsigned char)cache_lut[img_in[idx+(img_size/4)]];
	cache_img_buffer[threadIdx.x+2048] = (unsigned char)cache_lut[img_in[idx+2*(img_size/4)]];
	cache_img_buffer[threadIdx.x+3072] = (unsigned char)cache_lut[img_in[idx+3*(img_size/4)]];
	//__syncthreads();
	img_out[idx] = cache_img_buffer[threadIdx.x];
	img_out[idx+(img_size/4)] = cache_img_buffer[threadIdx.x+1024];
	img_out[idx+2*(img_size/4)] = cache_img_buffer[threadIdx.x+2048];
	img_out[idx+3*(img_size/4)] = cache_img_buffer[threadIdx.x+3072];
}



	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 33ms [1x1024])		//8 times shared memory and work/thread
__global__ void histogram_equalizationGPU7(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	int ix = threadIdx.x;	//col
	int idx = blockIdx.x*blockDim.x + ix;			//destination


	if(idx > img_size) 
		return;


	__shared__ int cache_lut[256];
	__shared__ unsigned char cache_img_buffer[8*1024];
							

	if(threadIdx.x <256)
		cache_lut[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();								


	cache_img_buffer[threadIdx.x] = (unsigned char)cache_lut[img_in[idx]];
	cache_img_buffer[threadIdx.x+1024] = (unsigned char)cache_lut[img_in[idx+(img_size/8)]];
	cache_img_buffer[threadIdx.x+2048] = (unsigned char)cache_lut[img_in[idx+2*(img_size/8)]];
	cache_img_buffer[threadIdx.x+3072] = (unsigned char)cache_lut[img_in[idx+3*(img_size/8)]];
	cache_img_buffer[threadIdx.x+4096] = (unsigned char)cache_lut[img_in[idx+4*(img_size/8)]];
	cache_img_buffer[threadIdx.x+5120] = (unsigned char)cache_lut[img_in[idx+5*(img_size/8)]];
	cache_img_buffer[threadIdx.x+6144] = (unsigned char)cache_lut[img_in[idx+6*(img_size/8)]];
	cache_img_buffer[threadIdx.x+7168] = (unsigned char)cache_lut[img_in[idx+7*(img_size/8)]];
	//__syncthreads();
	img_out[idx] = cache_img_buffer[threadIdx.x];
	img_out[idx+(img_size/8)] = cache_img_buffer[threadIdx.x+1024];
	img_out[idx+2*(img_size/8)] = cache_img_buffer[threadIdx.x+2048];
	img_out[idx+3*(img_size/8)] = cache_img_buffer[threadIdx.x+3072];
	img_out[idx+4*(img_size/8)] = cache_img_buffer[threadIdx.x+4096];
	img_out[idx+5*(img_size/8)] = cache_img_buffer[threadIdx.x+5120];
	img_out[idx+6*(img_size/8)] = cache_img_buffer[threadIdx.x+6144];
	img_out[idx+7*(img_size/8)] = cache_img_buffer[threadIdx.x+7168];
}


	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 33ms [1x1024])		//16 times shared memory and work/thread
__global__ void histogram_equalizationGPU8(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size)
{	

	int ix = threadIdx.x;	//col
	int idx = blockIdx.x*blockDim.x + ix;			//destination


	__shared__ int cache_lut[256];
	__shared__ unsigned char cache_img_buffer[16384];
							

	if(threadIdx.x <256)
		cache_lut[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();								


	cache_img_buffer[threadIdx.x] = (unsigned char)cache_lut[img_in[idx]];
	cache_img_buffer[threadIdx.x+1024] = (unsigned char)cache_lut[img_in[idx+(img_size/16)]];
	cache_img_buffer[threadIdx.x+2048] = (unsigned char)cache_lut[img_in[idx+2*(img_size/16)]];
	cache_img_buffer[threadIdx.x+3072] = (unsigned char)cache_lut[img_in[idx+3*(img_size/16)]];
	cache_img_buffer[threadIdx.x+4096] = (unsigned char)cache_lut[img_in[idx+4*(img_size/16)]];
	cache_img_buffer[threadIdx.x+5120] = (unsigned char)cache_lut[img_in[idx+5*(img_size/16)]];
	cache_img_buffer[threadIdx.x+6144] = (unsigned char)cache_lut[img_in[idx+6*(img_size/16)]];
	cache_img_buffer[threadIdx.x+7168] = (unsigned char)cache_lut[img_in[idx+7*(img_size/16)]];
	cache_img_buffer[threadIdx.x+8192] = (unsigned char)cache_lut[img_in[idx+8*(img_size/16)]];
	cache_img_buffer[threadIdx.x+9216] = (unsigned char)cache_lut[img_in[idx+9*(img_size/16)]];
	cache_img_buffer[threadIdx.x+10240] = (unsigned char)cache_lut[img_in[idx+10*(img_size/16)]];
	cache_img_buffer[threadIdx.x+11264] = (unsigned char)cache_lut[img_in[idx+11*(img_size/16)]];
	cache_img_buffer[threadIdx.x+12288] = (unsigned char)cache_lut[img_in[idx+12*(img_size/16)]];
	cache_img_buffer[threadIdx.x+13312] = (unsigned char)cache_lut[img_in[idx+13*(img_size/16)]];
	cache_img_buffer[threadIdx.x+14336] = (unsigned char)cache_lut[img_in[idx+14*(img_size/16)]];
	cache_img_buffer[threadIdx.x+15360] = (unsigned char)cache_lut[img_in[idx+15*(img_size/16)]];
	//__syncthreads();
	img_out[idx] = (unsigned char)cache_img_buffer[threadIdx.x];
	img_out[idx+(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+1024];
	img_out[idx+2*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+2048];
	img_out[idx+3*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+3072];
	img_out[idx+4*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+4096];
	img_out[idx+5*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+5120];
	img_out[idx+6*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+6144];
	img_out[idx+7*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+7168];
	img_out[idx+8*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+8192];
	img_out[idx+9*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+9216];
	img_out[idx+10*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+10240];
	img_out[idx+11*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+11264];
	img_out[idx+12*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+12288];
	img_out[idx+13*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+13312];
	img_out[idx+14*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+14336];
	img_out[idx+15*(img_size/16)] = (unsigned char)cache_img_buffer[threadIdx.x+15360];

}


	//just for the result image (WORKS FINE. TIME FOR 6400x6400 image =~ 33ms [1x1024])		//8 times shared memory and work/thread
__global__ void histogram_equalizationGPU9(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size, int numThreads)
{	

	int ix = threadIdx.x;	//col
	int idx = blockIdx.x*blockDim.x + ix;			//destination


	if(idx > img_size) 
		return;


	__shared__ int cache_lut[256];
	extern __shared__ unsigned char cache_img_buffer[];
							

	if(threadIdx.x <256)
		cache_lut[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();								


	cache_img_buffer[threadIdx.x] = (unsigned char)cache_lut[img_in[idx]];
	cache_img_buffer[threadIdx.x+numThreads] = (unsigned char)cache_lut[img_in[idx+(img_size/8)]];
	cache_img_buffer[threadIdx.x+2*numThreads] = (unsigned char)cache_lut[img_in[idx+2*(img_size/8)]];
	cache_img_buffer[threadIdx.x+3*numThreads] = (unsigned char)cache_lut[img_in[idx+3*(img_size/8)]];
	cache_img_buffer[threadIdx.x+4*numThreads] = (unsigned char)cache_lut[img_in[idx+4*(img_size/8)]];
	cache_img_buffer[threadIdx.x+5*numThreads] = (unsigned char)cache_lut[img_in[idx+5*(img_size/8)]];
	cache_img_buffer[threadIdx.x+6*numThreads] = (unsigned char)cache_lut[img_in[idx+6*(img_size/8)]];
	cache_img_buffer[threadIdx.x+7*numThreads] = (unsigned char)cache_lut[img_in[idx+7*(img_size/8)]];
	//__syncthreads();
	img_out[idx] = cache_img_buffer[threadIdx.x];
	img_out[idx+(img_size/8)] = cache_img_buffer[threadIdx.x+numThreads];
	img_out[idx+2*(img_size/8)] = cache_img_buffer[threadIdx.x+2*numThreads];
	img_out[idx+3*(img_size/8)] = cache_img_buffer[threadIdx.x+3*numThreads];
	img_out[idx+4*(img_size/8)] = cache_img_buffer[threadIdx.x+4*numThreads];
	img_out[idx+5*(img_size/8)] = cache_img_buffer[threadIdx.x+5*numThreads];
	img_out[idx+6*(img_size/8)] = cache_img_buffer[threadIdx.x+6*numThreads];
	img_out[idx+7*(img_size/8)] = cache_img_buffer[threadIdx.x+7*numThreads];
}













