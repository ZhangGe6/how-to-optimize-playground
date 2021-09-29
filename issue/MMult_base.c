// MMult_base.c
#include "params.h"
#include "MMult.h"

void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int j  = 0; j < n; ++j){
      for (int p = 0; p < k; ++p)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}