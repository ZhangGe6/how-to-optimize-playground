#include "params.h"
#include "MMult.h"
// save C_value in register and write the result back to cache only once

// slightly boost
// MMult_optim1_1 is not compatible here
void MMult_optim2_1(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
  register float C_accu;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_accu = 0;
      for (int p = 0; p < K; ++p){
        C_accu += A(i, p) * B(p, j); 
      }
      C(i, j) = C_accu;
    } 
  }
}
