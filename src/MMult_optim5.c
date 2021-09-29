#include "params.h"
#include "MMult.h"

#include "assert.h" 
#include <stdlib.h>
#include <stdio.h>

// Use pointer to reduce indexing overhead, as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_7
// Note that different from that repo (col major), this code is in row major fasion. 
// In that repo, for B, there is only one indexing for ij, compared with the original ij*4p
// Here there is p indexing for ij, is it avoidable?
// nearly no performance boost (compared with 3_2)
// TODO But the repo say that there is a big boost
void MMult_optim5_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double * pointer_b_p_j;

  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; ++p){
        pointer_b_p_j = &B(p, j);

        // TODO when B is very big, will miss cache here?
        C(i, j) += A(i, p) * *(pointer_b_p_j);          // &B(p, j)
        C(i, j + 1) += A(i, p) * *(pointer_b_p_j + 1);  // &B(p, j+1)
        C(i, j + 2) += A(i, p) * *(pointer_b_p_j + 2);  // &B(p, j+2)
        C(i, j + 3) += A(i, p) * *(pointer_b_p_j + 3);  // &B(p, j+3)
      }
    }
  }
}

// The unroll for inner `p for` is wierd in my machine (as in MMult_optim3.c MMult_optim3_5())
// so we do not do the unroll here as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_9


// Is it possible to indexing only once for one ij? (5_1 use p indexing) -> done
// This is similar to https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_9 by coincidence
// Use indirect addressing (for pointer_b_p_j here) to reduce the number of times the pointers need to be updated
// nearly no performance boost
void MMult_optim5_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double * pointer_b_p_j;

  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){
      pointer_b_p_j = &B(0, j);
      for (int p = 0; p < k; ++p){
        // TODO when B is very big, will miss cache here?
        C(i, j) += A(i, p) * *(pointer_b_p_j);          // &B(p, j)
        C(i, j + 1) += A(i, p) * *(pointer_b_p_j + 1);  // &B(p, j+1)
        C(i, j + 2) += A(i, p) * *(pointer_b_p_j + 2);  // &B(p, j+2)
        C(i, j + 3) += A(i, p) * *(pointer_b_p_j + 3);  // &B(p, j+3)

        pointer_b_p_j += ldb;
      }
    }
  }
}


