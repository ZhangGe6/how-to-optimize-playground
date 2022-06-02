#include "params.h"
#include "MMult.h"

// Routine for computing C = A * B
void MMult_base(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int p = 0; p < K; ++p)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}