#include "params.h"
#include "MMult.h"

// reorder the loops from i,j,k to i,k,j as suggested in https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/, a large boost
// It is the best reordering schema because all of A, B and C read/write in row-wise manner
void MMult_optim1_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int p  = 0; p < k; ++p){
      for (int j = 0; j < n; ++j)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}

// change the order from ijk to jik as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization1, but no performance boost
// I think the original author wrote like this because the original code is written in colomn-major fashion
// Here, B and C read/write in colomn-wise manner, but A reads in row-wise manner, so a better indexing order for colomn-major may be like MMult_optim1_3
void MMult_optim1_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int i  = 0; i < m; ++i){
        for (int p = 0; p < k; ++p)
            C(i, j) += A(i, p) * B(p, j);
    }
  }
}

// now all of A, B and C read/write in colomn-wise manner 
// a big performance degrade because all of A, B and C suffers cache miss
// JUST a illustration demo because this verision of code in written in row-major fashion
void MMult_optim1_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int p = 0; p < k; ++p){
        for (int i  = 0; i < m; ++i)
            C(i, j) += A(i, p) * B(p, j);
    }
  }
}
