
#include <stdio.h>

void MMult_benchmark(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

void MMult_base(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim1_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim2_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);