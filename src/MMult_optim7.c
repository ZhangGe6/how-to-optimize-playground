#include "params.h"
#include "MMult.h"
#include "assert.h" 

// https://github.com/flame/how-to-optimize-gemm/wiki#computing-a-4-x-4-block-of-c-at-a-time
// gonna to use vector registers

// use register to store B, too
// even slower. This is normal as https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8 illustrates,
// TODO: but why?
void MMult_optim7_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 4){
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

        // for 3rt row
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
void MMult_optim7_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int j = 0; j < n; j += 4){
    for (int i  = 0; i < m; i += 4){
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