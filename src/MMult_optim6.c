#include "params.h"
#include "MMult.h"
#include "assert.h" 


void MMult_optim6_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 4){
      for (int p = 0; p < k; ++p){
        // for 1st row
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);

        // for 2nd row
        C(i + 1, j) += A(i + 1, p) * B(p, j);
        C(i + 1, j + 1) += A(i + 1, p) * B(p, j + 1);
        C(i + 1, j + 2) += A(i + 1, p) * B(p, j + 2);
        C(i + 1, j + 3) += A(i + 1, p) * B(p, j + 3);

        // for 3rt row
        C(i + 2, j) += A(i + 2, p) * B(p, j);
        C(i + 2, j + 1) += A(i + 2, p) * B(p, j + 1);
        C(i + 2, j + 2) += A(i + 2, p) * B(p, j + 2);
        C(i + 2, j + 3) += A(i + 2, p) * B(p, j + 3);

        // for 4th row
        C(i + 3, j) += A(i + 3, p) * B(p, j);
        C(i + 3, j + 1) += A(i + 3, p) * B(p, j + 1);
        C(i + 3, j + 2) += A(i + 3, p) * B(p, j + 2);
        C(i + 3, j + 3) += A(i + 3, p) * B(p, j + 3);
      }
    }
  }
}


void MMult_optim6_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 1){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; p += 2){
        C(i, j) += A(i, p) * B(p, j);
        C(i, j) += A(i, p + 1) * B(p + 1, j);
      }
    }
  }
}
