#include "params.h"
#include "MMult.h"

#include "assert.h" 
#include <stdlib.h>
#include <stdio.h>

// Use pointer to reduce indexing overhead, as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_7
// So what is the reason that pointer can reduce indexing overhead? I GUESS that by using pointer, the calculation of index is simplified (pure addition and no muplication)
// Note that different from that repo (col major), this code is in row major fashion. 

// Use MMul_optim3_2 as baseline
// For B, there are totally `m(n/4)k` indexing, compared with the orignal `mnk` indexing
// nearly no performance boost
void MMult_optim5_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double * pointer_b_p_j;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      for (int p = 0; p < k; ++p){
        pointer_b_p_j = &B(p, j);

        C(i, j) += A(i, p) * *(pointer_b_p_j);          // &B(p, j)
        C(i, j + 1) += A(i, p) * *(pointer_b_p_j + 1);  // &B(p, j+1)
        C(i, j + 2) += A(i, p) * *(pointer_b_p_j + 2);  // &B(p, j+2)
        C(i, j + 3) += A(i, p) * *(pointer_b_p_j + 3);  // &B(p, j+3)
      }
    }
  }
}

// Is it possible to indexing only once for one ij? (5_1 use p indexing) -> done
// For B, there are totally `m(n/4)` indexing, compared with the orignal `mnk` indexing
// no performace boost
void MMult_optim5_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double * pointer_b_p_j;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      pointer_b_p_j = &B(0, j);
      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * *(pointer_b_p_j);          // &B(p, j)
        C(i, j + 1) += A(i, p) * *(pointer_b_p_j + 1);  // &B(p, j+1)
        C(i, j + 2) += A(i, p) * *(pointer_b_p_j + 2);  // &B(p, j+2)
        C(i, j + 3) += A(i, p) * *(pointer_b_p_j + 3);  // &B(p, j+3)

        pointer_b_p_j += ldb;
      }
    }
  }
}
