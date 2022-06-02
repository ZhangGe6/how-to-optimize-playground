#include <stdio.h>

// naive version
void MMult_base(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// change loop order
void MMult_optim1_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim1_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// save C_value in register and write the result back to cache only once
void MMult_optim2_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// unroll
void MMult_optim3_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim3_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim3_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim3_4(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim3_5(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// use more register
void MMult_optim4_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim4_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim4_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim4_4(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim4_1_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim4_2_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// use pointer to reduce indexing overhead
void MMult_optim5_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// use [vector] registers (SIMD) __m128
void MMult_optim6_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim6_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim6_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// use wider [vector] registers (SIMD) __m256
void MMult_optim7_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim7_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim7_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// cache blocking
void MMult_optim8_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim8_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
// void MMult_optim8_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
// void MMult_optim8_4(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// packing
void MMult_optim9_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);