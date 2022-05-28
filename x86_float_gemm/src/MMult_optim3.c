#include "params.h"
#include "MMult.h"

// unroll
// about why this helps:  https://www.sciencedirect.com/topics/computer-science/loop-unrolling

// unroll 1x4
// slightly performance boost
void MMult_optim3_1(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; ++j) {
      for (int p = 0; p < K; ++p) {
        C(i, j) += A(i, p) * B(p, j); 
        C(i + 1, j) += A(i + 1, p) * B(p, j); 
        C(i + 2, j) += A(i + 2, p) * B(p, j); 
        C(i + 3, j) += A(i + 3, p) * B(p, j); 
      }
    }
  }
}

void MMult_optim3_2(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 4) {
      for (int p = 0; p < K; ++p) {
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);
      }
    }
  }
}

void MMult_optim3_3(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int p = 0; p < K; p += 4) {
        C(i, j) += A(i, p) * B(p, j);
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        C(i, j) += A(i, p + 2) * B(p + 2, j);
        C(i, j) += A(i, p + 3) * B(p + 3, j);
      }   
    }
  }
}

// unroll 4x4
// further (and bigger) performace boost than unroll 1x4
void MMult_optim3_4(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      for (int p = 0; p < K; ++p) {
        // 1st col
        C(i, j) += A(i, p) * B(p, j); 
        C(i + 1, j) += A(i + 1, p) * B(p, j); 
        C(i + 2, j) += A(i + 2, p) * B(p, j); 
        C(i + 3, j) += A(i + 3, p) * B(p, j); 

        // 2nd col
        C(i, j + 1) += A(i, p) * B(p, j + 1); 
        C(i + 1, j + 1) += A(i + 1, p) * B(p, j + 1); 
        C(i + 2, j + 1) += A(i + 2, p) * B(p, j + 1); 
        C(i + 3, j + 1) += A(i + 3, p) * B(p, j + 1); 

        // 3rd col
        C(i, j + 2) += A(i, p) * B(p, j + 2); 
        C(i + 1, j + 2) += A(i + 1, p) * B(p, j + 2); 
        C(i + 2, j + 2) += A(i + 2, p) * B(p, j + 2); 
        C(i + 3, j + 2) += A(i + 3, p) * B(p, j + 2); 

        // 4th col
        C(i, j + 3) += A(i, p) * B(p, j + 3); 
        C(i + 1, j + 3) += A(i + 1, p) * B(p, j + 3); 
        C(i + 2, j + 3) += A(i + 2, p) * B(p, j + 3); 
        C(i + 3, j + 3) += A(i + 3, p) * B(p, j + 3); 
      }
    }
  }
}

// #pragma unroll
// the fastest one in MMult_optim3_*
void MMult_optim3_5(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int p = 0; p < K; ++p) {
      #pragma unroll
      for (int j = 0; j < N; ++j)
          C(i, j) += A(i, p) * B(p, j); 
    }
  }
}
