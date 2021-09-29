#include "params.h"
#include "MMult.h"

// Note that AddDot_2() is the same as AddDot() in MMult_optim1.c
// this new name is to avoid redefinition error when compiling

// unroll as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization2 suggested. 
// NO performance boost
void AddDot_2(int k, double *cur_A_row_starter, double *cur_B_col_starter, int ldb, double *cur_C);

void MMult_optim2_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){
        AddDot_2(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
        AddDot_2(k, &A(i, 0), &B(0, j + 1), ldb, &C(i, j + 1));
        AddDot_2(k, &A(i, 0), &B(0, j + 2), ldb, &C(i, j + 2));
        AddDot_2(k, &A(i, 0), &B(0, j + 3), ldb, &C(i, j + 3));
    }
  }
}

// NO performance boost, too
void MMult_optim2_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 10){
    for (int i  = 0; i < m; i += 1){
        AddDot_2(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
        AddDot_2(k, &A(i, 0), &B(0, j + 1), ldb, &C(i, j + 1));
        AddDot_2(k, &A(i, 0), &B(0, j + 2), ldb, &C(i, j + 2));
        AddDot_2(k, &A(i, 0), &B(0, j + 3), ldb, &C(i, j + 3));
        AddDot_2(k, &A(i, 0), &B(0, j + 4), ldb, &C(i, j + 4));
        AddDot_2(k, &A(i, 0), &B(0, j + 5), ldb, &C(i, j + 5));
        AddDot_2(k, &A(i, 0), &B(0, j + 6), ldb, &C(i, j + 6));
        AddDot_2(k, &A(i, 0), &B(0, j + 7), ldb, &C(i, j + 7));
        AddDot_2(k, &A(i, 0), &B(0, j + 8), ldb, &C(i, j + 8));
        AddDot_2(k, &A(i, 0), &B(0, j + 9), ldb, &C(i, j + 9));
    }
  }
}

void AddDot_2(int k, double *cur_A_row_starter, double *cur_B_col_starter, int ldb, double *cur_C){
    for(int p = 0; p < k; ++p){
        *cur_C += cur_A_row_starter[p] * cur_B_col_starter[p * ldb];
    }
}