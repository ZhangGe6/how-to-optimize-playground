// MMult_unroll.c
#include "params.h"
#include "MMult.h"
#include <stdio.h>

// This gives a correct calculation result
void MMult_unroll(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 2){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
      }
    }
  }
}

// However, when I tried inner loop, i or p, a wrong result occurs
void MMult_unroll_inner(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 1){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; p += 2){

        C(i, j) += A(i, p) * B(p, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p, A(i, p), p, j, B(p, j));
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p+1, A(i, p+1), p+1, j, B(p+1, j));
        
      }
      printf("\n");
    }
  }
}
