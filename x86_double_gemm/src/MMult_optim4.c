#include "params.h"
#include "MMult.h"

// start use register to reduce traffic between cache and registers as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_6 suggested
// As https://www.geeksforgeeks.org/understanding-register-keyword/ says, Registers are faster than memory to access, 
// so the variables which are most frequently used in a C program can be put in registers using register keyword

// use register based on MMult_optim3_2
// WE CAN SEE A LARGE SPEED BOOST!
void MMult_optim4_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_c_i_j, reg_c_i_j_add_1, reg_c_i_j_add_2, reg_c_i_j_add_3;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      // C(i, j) ~ C(i, j + 3) are frequently acceesd (in the loop), so we put it into register.
      reg_c_i_j = (double) 0;
      reg_c_i_j_add_1 = (double) 0;
      reg_c_i_j_add_2 = (double) 0;
      reg_c_i_j_add_3 = (double) 0;

      for (int p = 0; p < k; ++p){
        reg_c_i_j += A(i, p) * B(p, j);
        reg_c_i_j_add_1 += A(i, p) * B(p, j + 1);
        reg_c_i_j_add_2 += A(i, p) * B(p, j + 2);
        reg_c_i_j_add_3 += A(i, p) * B(p, j + 3);
      }
      C(i, j) += reg_c_i_j;
      C(i, j + 1) += reg_c_i_j_add_1;
      C(i, j + 2) += reg_c_i_j_add_2;
      C(i, j + 3) += reg_c_i_j_add_3;
    }
  }
}

void MMult_optim4_1_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_c_i_j, reg_c_i1_j, reg_c_i2_j, reg_c_i3_j;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 1){
      // C(i, j) ~ C(i, j + 3) are frequently acceesd (in the loop), so we put it into register.
      reg_c_i_j = (double) 0;
      reg_c_i1_j = (double) 0;
      reg_c_i2_j = (double) 0;
      reg_c_i3_j = (double) 0;

      for (int p = 0; p < k; ++p){
        reg_c_i_j += A(i, p) * B(p, j);
        reg_c_i1_j += A(i + 1, p) * B(p, j);
        reg_c_i2_j += A(i + 2, p) * B(p, j);
        reg_c_i3_j += A(i + 3, p) * B(p, j);
      }
      C(i, j) += reg_c_i_j;
      C(i + 1, j) += reg_c_i1_j;
      C(i + 2, j) += reg_c_i2_j;
      C(i + 3, j) += reg_c_i3_j;
    }
  }
}

// What about B(p, i) ~ B(p, j + 3)?
// slightly slower than (or nearly the same) MMult_optim3_2
// I think it is because of the PURELY EXTRA data move from cache to register
// In addition, I guess that it is super fast to move data from cache to register
void MMult_optim4_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_b_i_j, reg_b_i_j_add_1, reg_b_i_j_add_2, reg_b_i_j_add_3;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      for (int p = 0; p < k; ++p){
        // B(p, i) ~ B(p, j + 3) are frequently acceesd (in the loop), so we put it into register.
        // TODO But will moving to a register will cost more time? Maybe it is a trade-off
        reg_b_i_j = B(p, j);
        reg_b_i_j_add_1 = B(p, j + 1);
        reg_b_i_j_add_2 = B(p, j + 2);
        reg_b_i_j_add_3 = B(p, j + 3);

        C(i, j) += A(i, p) * reg_b_i_j;
        C(i, j + 1) += A(i, p) * reg_b_i_j_add_1;
        C(i, j + 2) += A(i, p) * reg_b_i_j_add_2;
        C(i, j + 3) += A(i, p) * reg_b_i_j_add_3;
      }
    }
  }
}

// A(i, p) is also frequently acceesd, How about move A(i, p) into register
// slightly faster than MMult_optim3_2
// I think it is because of 
  // 1. the EXTRA data move from cache to register        -> slower
  // 2. the reg_a_i_p is moved once and used for 4 times  -> faster
void MMult_optim4_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_a_i_p;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      for (int p = 0; p < k; ++p){
        // A(i, p) is frequently acceesd (in the loop), so we put it into register.
        // TODO But will moving to a register will cost more time? Maybe it is a trade-off
        reg_a_i_p = A(i, p);

        C(i, j) += reg_a_i_p * B(p, j);
        C(i, j + 1) += reg_a_i_p * B(p, j + 1);
        C(i, j + 2) += reg_a_i_p * B(p, j + 2);
        C(i, j + 3) += reg_a_i_p * B(p, j + 3);
      }
    }
  }
}


// How about make more use of register? i.e., move all C(i, j) ~ C(i, j + 3), B(p, i) ~ B(p, j + 3) and A(i, p) into register
// marginal boost (very small) compared to 4_1
void MMult_optim4_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_c_i_j, reg_c_i_j_add_1, reg_c_i_j_add_2, reg_c_i_j_add_3, \
                  reg_b_p_j, reg_b_p_j_add_1, reg_b_p_j_add_2, reg_b_p_j_add_3, \
                  reg_a_i_p;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      reg_c_i_j = (double) 0;
      reg_c_i_j_add_1 = (double) 0;
      reg_c_i_j_add_2 = (double) 0;
      reg_c_i_j_add_3 = (double) 0;

      for (int p = 0; p < k; ++p){
        reg_a_i_p = A(i, p);

        reg_b_p_j = B(p, j);
        reg_b_p_j_add_1 = B(p, j + 1);
        reg_b_p_j_add_2 = B(p, j + 2);
        reg_b_p_j_add_3 = B(p, j + 3);

        reg_c_i_j += reg_a_i_p * reg_b_p_j;
        reg_c_i_j_add_1 += reg_a_i_p * reg_b_p_j_add_1;
        reg_c_i_j_add_2 += reg_a_i_p * reg_b_p_j_add_2;
        reg_c_i_j_add_3 += reg_a_i_p * reg_b_p_j_add_3;
      }
      C(i, j) += reg_c_i_j;
      C(i, j + 1) += reg_c_i_j_add_1;
      C(i, j + 2) += reg_c_i_j_add_2;
      C(i, j + 3) += reg_c_i_j_add_3;
    }
  }
}

// Move only C(i, j+x) and A(i, p) to registers
// nearly the same with 4_1
void MMult_optim4_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double reg_c_i_j, reg_c_i_j_add_1, reg_c_i_j_add_2, reg_c_i_j_add_3, reg_a_i_p;

  for (int i = 0; i < m; i += 1){
    for (int j = 0; j < n; j += 4){
      // C(i, j) ~ C(i, j + 3) are frequently acceesd (in the loop), so we put it into register.
      reg_c_i_j = (double) 0;
      reg_c_i_j_add_1 = (double) 0;
      reg_c_i_j_add_2 = (double) 0;
      reg_c_i_j_add_3 = (double) 0;

      for (int p = 0; p < k; ++p){
        reg_a_i_p = A(i, p);

        reg_c_i_j += reg_a_i_p * B(p, j);
        reg_c_i_j_add_1 += reg_a_i_p * B(p, j + 1);
        reg_c_i_j_add_2 += reg_a_i_p * B(p, j + 2);
        reg_c_i_j_add_3 += reg_a_i_p * B(p, j + 3);
      }
      C(i, j) += reg_c_i_j;
      C(i, j + 1) += reg_c_i_j_add_1;
      C(i, j + 2) += reg_c_i_j_add_2;
      C(i, j + 3) += reg_c_i_j_add_3;
    }
  }
}