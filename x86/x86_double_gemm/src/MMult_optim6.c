#include "params.h"
#include "MMult.h"
#include "assert.h" 

// https://github.com/flame/how-to-optimize-gemm/wiki#computing-a-4-x-4-block-of-c-at-a-time
// start to computing a 4 x 4 block of C at a time

// naive one
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_5
// a further boost compared with MMul_optim3_2 (more levels of unrooling helps)
void MMult_optim6_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
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

        // for 3rd row
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

// use registers to store C
// large boost compared with MMult_optim6_1
// registers helps!
void MMult_optim6_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg;
                  // a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      for (int p = 0; p < k; ++p){
        // for 1st row
        c_00_reg += A(i, p) * B(p, j);
        c_01_reg += A(i, p) * B(p, j + 1);
        c_02_reg += A(i, p) * B(p, j + 2);
        c_03_reg += A(i, p) * B(p, j + 3);

        // for 2nd row
        c_10_reg += A(i + 1, p) * B(p, j);
        c_11_reg += A(i + 1, p) * B(p, j + 1);
        c_12_reg += A(i + 1, p) * B(p, j + 2);
        c_13_reg += A(i + 1, p) * B(p, j + 3);

        // for 3rd row
        c_20_reg += A(i + 2, p) * B(p, j);
        c_21_reg += A(i + 2, p) * B(p, j + 1);
        c_22_reg += A(i + 2, p) * B(p, j + 2);
        c_23_reg += A(i + 2, p) * B(p, j + 3);

        // for 4th row
        c_30_reg += A(i + 3, p) * B(p, j);
        c_31_reg += A(i + 3, p) * B(p, j + 1);
        c_32_reg += A(i + 3, p) * B(p, j + 2);
        c_33_reg += A(i + 3, p) * B(p, j + 3);
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}

// use registers to store C and A
// nearly no boost, even slower when the matrix size is small
void MMult_optim6_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      for (int p = 0; p < k; ++p){
        a_0p_reg = A(i, p); 
        a_1p_reg = A(i + 1, p);
        a_2p_reg = A(i + 2, p); 
        a_3p_reg = A(i + 3, p);

        // for 1st row
        c_00_reg += a_0p_reg * B(p, j);
        c_01_reg += a_0p_reg * B(p, j + 1);
        c_02_reg+= a_0p_reg * B(p, j + 2);
        c_03_reg += a_0p_reg * B(p, j + 3);

        // for 2nd row
        c_10_reg += a_1p_reg * B(p, j);
        c_11_reg += a_1p_reg * B(p, j + 1);
        c_12_reg += a_1p_reg * B(p, j + 2);
        c_13_reg += a_1p_reg * B(p, j + 3);

        // for 3rd row
        c_20_reg += a_2p_reg * B(p, j);
        c_21_reg += a_2p_reg * B(p, j + 1);
        c_22_reg += a_2p_reg * B(p, j + 2);
        c_23_reg += a_2p_reg * B(p, j + 3);

        // for 4th row
        c_30_reg += a_3p_reg * B(p, j);
        c_31_reg += a_3p_reg * B(p, j + 1);
        c_32_reg += a_3p_reg * B(p, j + 2);
        c_33_reg += a_3p_reg * B(p, j + 3);
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}

// use registers to store C and A, use pointers for B (in inner loop)
// nearly no boost
void MMult_optim6_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;
  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      for (int p = 0; p < k; ++p){
        a_0p_reg = A(i, p); 
        a_1p_reg = A(i + 1, p);
        a_2p_reg = A(i + 2, p);
        a_3p_reg = A(i + 3, p);

        b_p0_pntr = &B(p, j);
        b_p1_pntr = &B(p, j + 1);
        b_p2_pntr = &B(p, j + 2);
        b_p3_pntr = &B(p, j + 3);

        // for 1st row
        c_00_reg += a_0p_reg * *b_p0_pntr;
        c_01_reg += a_0p_reg * *b_p1_pntr;
        c_02_reg += a_0p_reg * *b_p2_pntr;
        c_03_reg += a_0p_reg * *b_p3_pntr;

        // for 2nd row
        c_10_reg += a_1p_reg * *b_p0_pntr;
        c_11_reg += a_1p_reg * *b_p1_pntr;
        c_12_reg += a_1p_reg * *b_p2_pntr;
        c_13_reg += a_1p_reg * *b_p3_pntr;

        // for 3rd row
        c_20_reg += a_2p_reg * *b_p0_pntr;
        c_21_reg += a_2p_reg * *b_p1_pntr;
        c_22_reg += a_2p_reg * *b_p2_pntr;
        c_23_reg += a_2p_reg * *b_p3_pntr;

        // for 4th row
        c_30_reg += a_3p_reg * *b_p0_pntr;
        c_31_reg += a_3p_reg * *b_p1_pntr;
        c_32_reg += a_3p_reg * *b_p2_pntr;
        c_33_reg += a_3p_reg * *b_p3_pntr;
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}

// use register to store all of A, B and C
// Use 4x4 registers to store current C block
// Uss 4x1 registers to store colomn of A block
// Uss 1x4 registers to store row of B block
// even slower. This is normal as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8 illustrates,
// TODO: but why?
void MMult_optim6_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,

                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      b_p0_pntr = &B(0, j);
      b_p1_pntr = &B(0, j + 1);
      b_p2_pntr = &B(0, j + 2); 
      b_p3_pntr = &B(0, j + 3);

      for (int p = 0; p < k; ++p){
        a_0p_reg = A(i, p); 
        a_1p_reg = A(i + 1, p);
        a_2p_reg = A(i + 2, p);
        a_3p_reg = A(i + 3, p);

        b_p0_reg = *b_p0_pntr;
        b_p1_reg = *b_p1_pntr;
        b_p2_reg = *b_p2_pntr;
        b_p3_reg = *b_p3_pntr;

        // for 1st row
        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;
        c_02_reg += a_0p_reg * b_p2_reg;
        c_03_reg += a_0p_reg * b_p3_reg;

        // for 2nd row
        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;
        c_12_reg += a_1p_reg * b_p2_reg;
        c_13_reg += a_1p_reg * b_p3_reg;

        // for 3rd row
        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;
        c_22_reg += a_2p_reg * b_p2_reg;
        c_23_reg += a_2p_reg * b_p3_reg;

        // for 4th row
        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;
        c_32_reg += a_3p_reg * b_p2_reg;
        c_33_reg += a_3p_reg * b_p3_reg;

        // update b_px_pntr
        b_p0_pntr += ldb;
        b_p1_pntr += ldb;
        b_p2_pntr += ldb; 
        b_p3_pntr += ldb;
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}


// rearrange as https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_9.c 
// so that we compute C block two rows at a time.
// no perforamce boost compared with MMul_optim6_5 (before rearranging)
void MMult_optim6_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      b_p0_pntr = &B(0, j);
      b_p1_pntr = &B(0, j + 1);
      b_p2_pntr = &B(0, j + 2); 
      b_p3_pntr = &B(0, j + 3);

      for (int p = 0; p < k; ++p){
        a_0p_reg = A(i, p); 
        a_1p_reg = A(i + 1, p);
        a_2p_reg = A(i + 2, p);
        a_3p_reg = A(i + 3, p);

        b_p0_reg = *b_p0_pntr;
        b_p1_reg = *b_p1_pntr;
        b_p2_reg = *b_p2_pntr;
        b_p3_reg = *b_p3_pntr;

        /* First row and second rows */
        c_00_reg += a_0p_reg * b_p0_reg;
        c_10_reg += a_1p_reg * b_p0_reg;

        c_01_reg += a_0p_reg * b_p1_reg;
        c_11_reg += a_1p_reg * b_p1_reg;

        c_02_reg += a_0p_reg * b_p2_reg;
        c_12_reg += a_1p_reg * b_p2_reg;

        c_03_reg += a_0p_reg * b_p3_reg;
        c_13_reg += a_1p_reg * b_p3_reg;

        /* Third and fourth rows */
        c_20_reg += a_2p_reg * b_p0_reg;
        c_30_reg += a_3p_reg * b_p0_reg;

        c_21_reg += a_2p_reg * b_p1_reg;
        c_31_reg += a_3p_reg * b_p1_reg;

        c_22_reg += a_2p_reg * b_p2_reg;
        c_32_reg += a_3p_reg * b_p2_reg;

        c_23_reg += a_2p_reg * b_p3_reg;
        c_33_reg += a_3p_reg * b_p3_reg;

        // update b_px_pntr
        b_p0_pntr += ldb;
        b_p1_pntr += ldb;
        b_p2_pntr += ldb; 
        b_p3_pntr += ldb;
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}

// Note that the arrangement in MMult_optim6_6 is served for colomn-major style like flame version,
// and to be compatible with row-major style here, we should re-write it
void MMult_optim6_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      b_p0_pntr = &B(0, j);
      b_p1_pntr = &B(0, j + 1);
      b_p2_pntr = &B(0, j + 2); 
      b_p3_pntr = &B(0, j + 3);

      for (int p = 0; p < k; ++p){
        a_0p_reg = A(i, p); 
        a_1p_reg = A(i + 1, p);
        a_2p_reg = A(i + 2, p);
        a_3p_reg = A(i + 3, p);

        b_p0_reg = *b_p0_pntr;
        b_p1_reg = *b_p1_pntr;
        b_p2_reg = *b_p2_pntr;
        b_p3_reg = *b_p3_pntr;

        /* First colomns and second colomns */
        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;

        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;

        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;

        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;

        /* Third colomns fourth colomns */
        c_02_reg += a_0p_reg * b_p2_reg;
        c_03_reg += a_0p_reg * b_p3_reg;

        c_12_reg += a_1p_reg * b_p2_reg;
        c_13_reg += a_1p_reg * b_p3_reg;

        c_22_reg += a_2p_reg * b_p2_reg;
        c_23_reg += a_2p_reg * b_p3_reg;

        c_32_reg += a_3p_reg * b_p2_reg;
        c_33_reg += a_3p_reg * b_p3_reg;

        // update b_px_pntr
        b_p0_pntr += ldb;
        b_p1_pntr += ldb;
        b_p2_pntr += ldb; 
        b_p3_pntr += ldb;
      }
      C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;
    }
  }  
}