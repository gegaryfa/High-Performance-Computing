#ifndef __CUDA_ERROR_CHECKS__
// Define this to turn on error checking
#define __CUDA_ERROR_CHECKS__

#define CUDA_ERROR_CHECK
#define CudaSafeCall(err)	__cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()	__cudaCheckError(__FILE__, __LINE__)



void __cudaSafeCall (cudaError_t err, const char *file, const int line);
void __cudaCheckError (const char *file, const int line);

#endif /* __CUDA_ERROR_CHECKS__ */
