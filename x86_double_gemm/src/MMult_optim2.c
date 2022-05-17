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

// 循环展开的有效性原因，除了通常说的，降低分支预测，循环终止判断的开销外，
// 这篇：https://zhuanlan.zhihu.com/p/395020419， 提供了一个新的角度：让访存局部性比较差的元素（B中元素），一次读取后，进行多次操作
// （但在实验里，对i进行循环展开，并无加速效果）
// https://github.com/flame/blislab/blob/master/tutorial.pdf 这里说循环展开可以减少一些值的更新次数
// and here: https://stackoverflow.com/a/22279341/10096987