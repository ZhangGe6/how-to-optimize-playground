#include "params.h"
#include "utils.h"
#include "MMult.h"

// void MMult_benchmark(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
//     cublasHandle_t blas_handle;  
//     checkCuBlasErrors (cublasCreate(&blas_handle));
//     float alpha = 1.0, beta = 0.0;
//     checkCuBlasErrors (
//         // https://blog.csdn.net/u011197534/article/details/78378536
//         cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
//             m, n, k, &alpha, 
//             d_A, k, d_B, n, &beta, d_C, m
//         )
//     );
// }

void MMult_benchmark(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    float alpha = 1.0, beta = 0.0;
    checkCuBlasErrors (
        // https://blog.csdn.net/u011197534/article/details/78378536
        cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            m, n, k, &alpha, 
            d_A, k, d_B, n, &beta, d_C, m
        )
    );
}