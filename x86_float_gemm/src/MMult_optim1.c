#include "params.h"
#include "MMult.h"

// reorder the loops from i,j,k to i,k,j as suggested in https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
// a large boost
// It is the best reordering schema because all of A, B and C read/write in row-wise manner (least cashe miss)
void MMult_optim1_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  for (int i = 0; i < M; ++i) {
    for (int p = 0; p < K; ++p) {
      for (int j = 0; j < N; ++j)
          C(i, j) += A(i, p) * B(p, j); 
    }
  }
}

// a big performance degrade because all of A, B and C suffers cache miss
void MMult_optim1_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  
  for (int j = 0; j < N; ++j) {
    for (int p = 0; p < K; ++p) {
      for (int i = 0; i < M; ++i)
          C(i, j) += A(i, p) * B(p, j); 
    }
  }
}
