#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H


typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



extern PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

//CPU Kernels
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

void  create_lut(int * hist_in, int img_size, int nbr_bin, int **lut);

//GPU Kernels
__global__ 	void histogramGPU_1(int *hist_out, unsigned char *img_in, int img_size);
__global__  void histogramGPU_2(int* hist_out, unsigned char *img_in, int img_size);
__global__ void histogram_equalizationGPU1(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU2(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU3(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);

__global__ void histogram_equalizationGPU5(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU6(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU7(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU8(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU9(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size, int numthreads);
__global__ void histogram_equalizationGPU10(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size);
__global__ void histogram_equalizationGPU11(unsigned char *img_out, unsigned char *img_in,
                                 int *lut, int img_size, int numThreads);


//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif

//bash colors
#define ANSI_COLOR_WHITE   "\x1B[0m"	//WHITE
#define ANSI_COLOR_GREEN   "\x1b[32m"   //GREEN color for GPU messages :)
#define ANSI_COLOR_RED     "\x1b[31m"	//RED
#define ANSI_COLOR_YELLOW  "\x1b[33m"	//YELLOW


/*Define in case of using the constant memory kernel*/
//#define USE_CONSTANT

/* 
 * If TIMING is defined, CPU & GPU run their kernels
 * 12 times, avoid max and min times,
 * and compute an average computation time
 */
#define TIMING


