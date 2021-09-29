#include "params.h"
#include "MMult.h"
#include "assert.h" 

// TODO get the wrong result
void MMult_optim6_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 4){
      for (int p = 0; p < k; ++p){
        // for 1st row
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);

        // for 2nd row
        C(i + 1, j) += A(i + 1, p) * B(p, j);
        C(i + 1, j + 1) += A(i + 1, p) * B(p, j + 1);
        C(i + 1, j + 2) += A(i + 1, p) * B(p, j + 2);
        C(i + 1, j + 3) += A(i + 1, p) * B(p, j + 3);

        // for 3rt row
        C(i + 2, j) += A(i + 2, p) * B(p, j);
        C(i + 2, j + 1) += A(i + 2, p) * B(p, j + 1);
        C(i + 2, j + 2) += A(i + 2, p) * B(p, j + 2);
        C(i + 2, j + 3) += A(i + 2, p) * B(p, j + 3);

        // for 4th row
        C(i + 3, j) += A(i + 3, p) * B(p, j);
        C(i + 3, j + 1) += A(i + 3, p) * B(p, j + 1);
        C(i + 3, j + 2) += A(i + 3, p) * B(p, j + 2);
        C(i + 3, j + 3) += A(i + 3, p) * B(p, j + 3);
      }
    }
  }
}

// TODO So Weird AGAIN (like optim3_3)! When I unroll the outer loop like j, it's OK, but when I unroll the inner loop like i or p, 
// the caculation result goes wrong. It seems that the indexed number (like B(p+1) in the follow one) does not update in time. 
void MMult_optim6_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 1){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; p += 2){
        
        C(i, j) += A(i, p) * B(p, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p, A(i, p), p, j, B(p, j));
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p+1, A(i, p+1), p+1, j, B(p+1, j));
        // printf("  And B(3, 0) is should be %f\n", B(3, 0));
        // if (p == 3 && j ==0){
        //   assert(B(p, j) == B(3, 0));
        // }
      }
      printf("\n");
    }
  }
}
