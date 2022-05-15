#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


void allocate_space(int m, int k, int n, float *matA, float *matB, float *matC);
void random_matrix( int m, int n, float *mat, int ldm);
void zero_matrix( int m, int n, float *mat, int ldm);
void print_matrix(int m, int n, float *mat, int ldm);
float compare_matrix(int m, int n, float *mat1, float *mat2, int ldm);
void print_gpu_info();


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
// __m128d is a data type that the compiler will hopefully store in a XMM 128 bit register when optimizing
// https://stackoverflow.com/questions/53757633/what-is-m128d
// https://zhuanlan.zhihu.com/p/55327037
// 2 float, 4 float, ...
// union: https://www.runoob.com/cprogramming/c-unions.html
typedef union {
    __m128d reg;    
    float value[2];
} v2d_regv;


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
