#include "params.h"
#include "MMult.h"

// reorder the loops from i,j,k to i,k,j as suggested in https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/, a large boost
void MMult_optim1_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int p  = 0; p < k; ++p){
      for (int j = 0; j < n; ++j)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}

//  change the order from ijk to jik as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization1, but no performance boost
void MMult_optim1_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int i  = 0; i < m; ++i){
        for (int p = 0; p < k; ++p)
            C(i, j) += A(i, p) * B(p, j);
    }
  }
}

// change the order from ijk to jik and use a inner function(AddDot) fasion as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization1, but no performance boost
void AddDot_optim1(int k, double *cur_A_row_starter, double *cur_B_col_starter, int ldb, double *cur_C);

void MMult_optim1_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int i  = 0; i < m; ++i){
        AddDot_optim1(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
    }
  }
}

void AddDot_optim1(int k, double *cur_A_row_starter, double *cur_B_col_starter, int ldb, double *cur_C){
    for(int p = 0; p < k; ++p){
        *cur_C += cur_A_row_starter[p] * cur_B_col_starter[p * ldb];
    }
}