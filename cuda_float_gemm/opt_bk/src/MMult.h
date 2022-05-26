
#include <stdio.h>

void MMult_benchmark(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

void MMult_base(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

void MMult_optim1_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

// use block
void MMult_optim2_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

// increase the work per thread
void MMult_optim3_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim3_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim3_3(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim3_4(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim3_5(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim3_3_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

// use wider datatype like float4
void MMult_optim4_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim4_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

// use more flexible block
void MMult_optim5_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);
void MMult_optim5_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc);

void MMult_optim6_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

void MMult_optim7_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim7_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim7_3(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim7_4(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

// try prefetch
void MMult_optim8_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim8_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim8_3(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
void MMult_optim8_4(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

void MMult_optim9_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

void MMult_optim10_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);