#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 

#include <iostream>
#include <random>
#include "params.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

void random_matrix(float *mat, int m, int n);
void zero_matrix(float *mat, int m, int n);
void print_matrix(float *mat, int m, int n);
float compare_matrix(float *mat1, float *mat2, int m, int n);
void print_gpu_info();


#define checkCudaErrors(func)		\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)			\
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

static const char *_cuBlasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

#define checkCuBlasErrors(func)	    \
{									\
    cublasStatus_t e = (func);		\
    if(e != CUBLAS_STATUS_SUCCESS)	\
        printf ("%s %d CuBlas: %s", __FILE__,  __LINE__, _cuBlasGetErrorEnum(e));		\
}


// https://stackoverflow.com/a/14038590/10096987
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}