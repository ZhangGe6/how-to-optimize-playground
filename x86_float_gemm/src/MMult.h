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

// use wider [vector] registers (SIMD) __m256
void MMult_optim7_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim7_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);

// cache blocking
void MMult_optim8_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
void MMult_optim8_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
// void MMult_optim8_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);
// void MMult_optim8_4(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc);



// packing




// use pointer to reduce the indexing overhead








// void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_base_k_seg(int m, int k_s, int k_e, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // change index order (loop reordering)
// void MMult_optim1_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim1_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim1_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim1_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim1_1_k_seg(int m, int k_s, int k_e, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to unroll
// // TODO: will +4 out of bound or missing the last few (1, 2, 3) col/rows? 
// // --> If the row or col % 4 == 0, then there will no problem. Otherwise some other codes are needed
// void MMult_optim2_0(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim2_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim2_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim2_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // 'caculate more' at a time (1x4)
// void MMult_optim3_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim3_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to use register
// void MMult_optim4_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim4_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim4_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim4_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim4_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to use pointer to reduce the indexing overhead
// void MMult_optim5_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim5_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to caculate 4x4 at a time 
// void MMult_optim6_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim6_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to use [vector] registers (SIMD)
// void MMult_optim7_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim7_1_k_seg(int m, int k_s, int k_e, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start to use cache blocking
// void MMult_optim8_0(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_5_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim8_8(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

// // start use packing
// void MMult_optim9_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_8(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_9(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
// void MMult_optim9_10(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);