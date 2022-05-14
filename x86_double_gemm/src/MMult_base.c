#include "params.h"
#include "MMult.h"

// Routine for computing C = A * B
void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int j = 0; j < n; ++j){
      for (int p = 0; p < k; ++p)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}

void MMult_base_k_seg(int m, int k_s, int k_e, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int j = 0; j < n; ++j){
      for (int p = k_s; p < k_e; ++p)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}