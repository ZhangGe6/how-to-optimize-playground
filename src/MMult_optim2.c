#include "params.h"
#include "MMult.h"

// a helper function for calculating a single elment in C
void AddDot(int k, double *cur_A_row_starter, double *cur_B_col_starter, int ldb, double *cur_C){
    for(int p = 0; p < k; ++p){
        *cur_C += cur_A_row_starter[p] * cur_B_col_starter[p * ldb];
    }
}

// a baseline using AddDot
// NO performance boost
void MMult_optim2_0(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        AddDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
    }
  }
}

// unroll as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization2 suggested. 
// NO performance boost
void MMult_optim2_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
        AddDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
        AddDot(k, &A(i, 0), &B(0, j + 1), ldb, &C(i, j + 1));
        AddDot(k, &A(i, 0), &B(0, j + 2), ldb, &C(i, j + 2));
        AddDot(k, &A(i, 0), &B(0, j + 3), ldb, &C(i, j + 3));
    }
  }
}

// NO performance boost, too
void MMult_optim2_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 10){
        AddDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
        AddDot(k, &A(i, 0), &B(0, j + 1), ldb, &C(i, j + 1));
        AddDot(k, &A(i, 0), &B(0, j + 2), ldb, &C(i, j + 2));
        AddDot(k, &A(i, 0), &B(0, j + 3), ldb, &C(i, j + 3));
        AddDot(k, &A(i, 0), &B(0, j + 4), ldb, &C(i, j + 4));
        AddDot(k, &A(i, 0), &B(0, j + 5), ldb, &C(i, j + 5));
        AddDot(k, &A(i, 0), &B(0, j + 6), ldb, &C(i, j + 6));
        AddDot(k, &A(i, 0), &B(0, j + 7), ldb, &C(i, j + 7));
        AddDot(k, &A(i, 0), &B(0, j + 8), ldb, &C(i, j + 8));
        AddDot(k, &A(i, 0), &B(0, j + 9), ldb, &C(i, j + 9));
    }
  }
}

// NO performance boost
void MMult_optim2_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 1){
        AddDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
        AddDot(k, &A(i + 1, 0), &B(0, j), ldb, &C(i + 1, j));
        AddDot(k, &A(i + 2, 0), &B(0, j), ldb, &C(i + 2, j));
        AddDot(k, &A(i + 3, 0), &B(0, j), ldb, &C(i + 3, j));
    }
  }
}
