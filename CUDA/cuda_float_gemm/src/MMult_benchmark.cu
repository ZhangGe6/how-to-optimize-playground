#include "params.h"
#include "utils.h"
#include "MMult.h"

// Note that cublasSgemm works in col-major, so we calculate by CT = BT * AT
// https://blog.csdn.net/u011197534/article/details/78378536
// https://blog.csdn.net/HaoBBNuanMM/article/details/103054357
void MMult_benchmark(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {
    float alpha = 1, beta = 0;
    
    checkCuBlasErrors (
        cublasSgemm(handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N, 
                    N,
                    M, 
                    K,
                    &alpha, 
                    B, 
                    N,
                    A,
                    K, 
                    &beta, 
                    C, 
                    N
        )
    );
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

}