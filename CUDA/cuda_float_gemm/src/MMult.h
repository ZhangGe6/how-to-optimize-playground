
#include <stdio.h>

// cublas benchmark
void MMult_benchmark(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// naive version
void MMult_base(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// use register to save Cvalue and write back to global memory at once
void MMult_optim1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// use block (square block now, block into shared memory)
void MMult_optim2_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// increase the work per thread
void MMult_optim3_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);
void MMult_optim3_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// try different ways of reading from global memory to shared memory 
void MMult_optim4_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// use wider datatype like float4
void MMult_optim5_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);
void MMult_optim5_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// block into register and compute
void MMult_optim6_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// square block optimization playground
void MMult_optim6_1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// use block of more flexible size (rectangle block)
void MMult_optim_rect1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);
void MMult_optim_rect1_1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);
void MMult_optim_rect1_1_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N);

// // try prefetch
// void MMult_optim8_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
// void MMult_optim8_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
// void MMult_optim8_3(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);
// void MMult_optim8_4(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

// void MMult_optim9_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);

// void MMult_optim10_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N, float alpha, float beta);