#include "params.h"
#include "MMult.h"

// compute 1x4 at a time and thorw away the AddDot() fasion, as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_4 suggested
// inline the four separate inner products 
// No speed boost again
void MMult_optim3_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){

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
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
        C(i, j + 2) += A(i, p) * B(p, j + 2);
        C(i, j + 3) += A(i, p) * B(p, j + 3);
      }
    }
  }
}


// try to unroll the inner `p for`, but some errors happen. 3_4 and 3_5 are used for debugging
void MMult_optim3_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 1){
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

void MMult_optim3_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int i  = 0; i < m; ++i){
      double prev_C_i_j = C(i, j);
      for (int p = 0; p < k; p += 2){
        C(i, j) += A(i, p) * B(p, j);
        // printf("j%d, i%d, p%d -> A%f, B%f, C%f\n", j, i, p, A(i, p), B(p, j), C(i, j));
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        // printf("j%d, i%d, p%d -> A%f, B%f, C%f\n", j, i, p+1, A(i, p + 1), B(p + 1, j), C(i, j));
        // C(i, j) += A(i, p + 2) * B(p + 2, j);
        // // printf("j%d, i%d, p%d -> A%f, B%f, C%f\n", j, i, p+2, A(i, p + 2), B(p + 2, j), C(i, j));
        // C(i, j) += A(i, p + 3) * B(p + 3, j);
        // printf("j%d, i%d, p%d -> A%f, B%f, C%f\n", j, i, p+3, A(i, p + 3), B(p + 3, j), C(i, j));
      }
      double final_C_i_j = C(i, j);
      // printf("C(%d, %d) change from %f, to %f\n", i, j, prev_C_i_j, final_C_i_j);
      
    }
    // printf("\n");
  }
  // printf("\n");
}

void MMult_optim3_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; ++j){
    for (int i  = 0; i < m; ++i){
      double no_unroll_prev_C_i_j = C(i, j);
      // no unroll
      printf("B(5, 0) = %f\n", B(5, 0));
      for (int p = 0; p < k; p += 1){
        C(i, j) += A(i, p) * B(p, j);
        printf("j%d, i%d, p%d -> A(%d %d)%f, B(%d %d)%f, C%f\n", j, i, p, i, p, A(i, p), p, j, B(p, j), C(i, j));
      }
      double no_unroll_final_C_i_j = C(i, j);
      printf("--> no unroll: C(%d, %d) change from %f, to %f\n", i, j, no_unroll_prev_C_i_j, no_unroll_final_C_i_j);
      printf("B(5, 0) = %f\n", B(5, 0));

      // unroll
      C(i, j) = 0; // reset
      double unroll_prev_C_i_j = C(i, j);
      for (int p = 0; p < k; p += 4){
        C(i, j) += A(i, p) * B(p, j);
        printf("j%d, i%d, p%d -> A(%d %d)%f, B(%d %d)%f, C%f\n", j, i, p, i, p,  A(i, p), p, j, B(p, j), C(i, j));
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        printf("B(5, 0) = %f, now p=%d, p+1=%d, B(%d, 0)=%f\n", B(5, 0), p, p+1, p+1, B(p+1, 0));  // TODO It's too buggy
        printf("j%d, i%d, p%d -> A(%d %d)%f, B(%d %d)%f, C%f\n", j, i, p+1, i, p+1,  A(i, p + 1), p+1, j, B(p + 1, j), C(i, j));
        C(i, j) += A(i, p + 2) * B(p + 2, j);
        printf("j%d, i%d, p%d -> A(%d %d)%f, B(%d %d)%f, C%f\n", j, i, p+2, i, p+2,  A(i, p + 2), p+2, j, B(p + 2, j), C(i, j));
        C(i, j) += A(i, p + 3) * B(p + 3, j);
        printf("j%d, i%d, p%d -> A(%d %d)%f, B(%d %d)%f, C%f\n", j, i, p+3, i, p+3,  A(i, p + 3), p+3, j, B(p + 3, j), C(i, j));
      }
      double unroll_final_C_i_j = C(i, j);
      printf("--> unroll: C(%d, %d) change from %f, to %f\n", i, j, unroll_prev_C_i_j, unroll_final_C_i_j);
      printf("B(5, 0) = %f\n", B(5, 0));


      // // no unroll again
      // C(i, j) = 0; // reset
      // no_unroll_prev_C_i_j = C(i, j);
      // for (int p = 0; p < k; p += 1){
      //   C(i, j) += A(i, p) * B(p, j);
      //   printf("j%d, i%d, p%d -> A%f, B%f, C%f\n", j, i, p, A(i, p), B(p, j), C(i, j));
      // }
      // no_unroll_final_C_i_j = C(i, j);
      // printf("--> no unroll: C(%d, %d) change from %f, to %f\n", i, j, no_unroll_prev_C_i_j, no_unroll_final_C_i_j);
      printf("\n");
    }
    
  }
}