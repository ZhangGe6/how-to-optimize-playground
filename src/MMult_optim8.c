#include "params.h"
#include "MMult.h"

// Use cache blocking 
// Not exactly the same with:
// https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_11.c

// How it works
// 1. https://www.youtube.com/watch?v=G92BCtfTwOE (at time 5:00)
// 2. https://stackoverflow.com/questions/63614160/how-does-cache-blocking-actually-speed-up-performance

int blockSize = 40;  // set to a divisor to `msize` in main.c for simplicity

// No performance gain compared with pure MMult_base
void MMult_optim8_0(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {

      for (int i = mBlockStart; i < mBlockStart + blockSize; ++i){
        for (int j = nBlockStart; j < nBlockStart + blockSize; ++j){
          for (int p = 0; p < k; ++p)
            C(i, j) += A(i, p) * B(p, j);
        }
      }

    }
  }
}

// No performance gain compared with pure MMult_base
void MMult_optim8_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      MMult_base(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
    }
  }
}

// even slower than MMult_optim1_1 (confusing)
void MMult_optim8_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      MMult_optim1_1(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
    }
  }
}

void MMult_optim8_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      MMult_optim7_1(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
    }
  }
}