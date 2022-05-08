#include "params.h"
#include "MMult.h"

// compute 1x4 at a time and thorw away the AddDot() fasion, as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_4 suggested
// inline the four separate inner products 
// No speed boost again
void MMult_optim3_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){

      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * B(p, j); // keep in mind that A(i, 0) is the row_starter of row i of A, and B(0, j) is the col_starter of col j of B
      }

      for (int p = 0; p < k; ++p){
        C(i, j + 1) += A(i, p) * B(p, j + 1);
      }

      for (int p = 0; p < k; ++p){
        C(i, j + 2) += A(i, p) * B(p, j + 2);
      }

      for (int p = 0; p < k; ++p){
        C(i, j + 3) += A(i, p) * B(p, j + 3);
      }

    }
  }
}

// fuse the loops into one, thereby computing the four inner products simultaneously in one loop, as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_5 suggested
// WE CAN SEE SOME SPEED BOOST!
void MMult_optim3_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);
      }
    }
  }
}


// try to continue to unroll the inner `k for`, DEGRADE
void MMult_optim3_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      for (int p = 0; p < k; p += 4){

        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);

        C(i, j) += A(i, p + 1) * B(p + 1, j);
        C(i, j + 1) += A(i, p + 1) * B(p + 1, j + 1);
        C(i, j + 2) += A(i, p + 1) * B(p + 1, j + 2);
        C(i, j + 3) += A(i, p + 1) * B(p + 1, j + 3);

        C(i, j) += A(i, p + 2) * B(p + 2, j);
        C(i, j + 1) += A(i, p + 2) * B(p + 2, j + 1);
        C(i, j + 2) += A(i, p + 2) * B(p + 2, j + 2);
        C(i, j + 3) += A(i, p + 2) * B(p + 2, j + 3);

        C(i, j) += A(i, p + 3) * B(p + 3, j);
        C(i, j + 1) += A(i, p + 3) * B(p + 3, j + 1);
        C(i, j + 2) += A(i, p + 3) * B(p + 3, j + 2);
        C(i, j + 3) += A(i, p + 3) * B(p + 3, j + 3);
      }
    }
  }
}


// unloop m, no difference between unrolling n
void MMult_optim3_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 4){
    for (int j  = 0; j < n; j += 1){
      for (int p = 0; p < k; ++p){

        C(i, j) += A(i, p) * B(p, j);
        C(i + 1, j) += A(i + 1, p) * B(p, j);
        C(i + 2, j) += A(i + 2, p) * B(p, j);
        C(i + 3, j) += A(i + 3, p) * B(p, j);
      }
    }
  }
}

// combine a faster reorder (MMult_optim1_1) and unloop
// Faster than MMult_optim1_1
void MMult_optim3_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int p = 0; p < k; ++p){
      for (int j = 0; j < n; j += 4) {
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);
      }
    }
  }
}
