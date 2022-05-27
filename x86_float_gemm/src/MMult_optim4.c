#include "params.h"
#include "MMult.h"

// use register to save the most frequently used data

// use register for the 1x4 unrolling
// performance boost !
void MMult_optim4_1(float *A, float *B, float *C, int M, int K, int N)
{
  register float reg_c_i_j, reg_c_i_j1, reg_c_i_j2, reg_c_i_j3;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 4) {
      reg_c_i_j = (float) 0;
      reg_c_i_j1 = (float) 0;
      reg_c_i_j2 = (float) 0;
      reg_c_i_j3 = (float) 0;
      
      for (int p = 0; p < K; ++p) {
        reg_c_i_j += A(i, p) * B(p, j);
        reg_c_i_j1 += A(i, p) * B(p, j + 1);
        reg_c_i_j2 += A(i, p) * B(p, j + 2);
        reg_c_i_j3 += A(i, p) * B(p, j + 3);
      }
      C(i, j) = reg_c_i_j;
      C(i, j + 1) = reg_c_i_j1;
      C(i, j + 2) = reg_c_i_j2;
      C(i, j + 3) = reg_c_i_j3;
    }
  }
}

// use register for the 4x4 unrolling
void MMult_optim4_2(float *A, float *B, float *C, int M, int K, int N)
{
  register float  c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      for (int p = 0; p < K; ++p) {
        // 1st col
        c_00_reg += A(i, p) * B(p, j); 
        c_10_reg += A(i + 1, p) * B(p, j); 
        c_20_reg += A(i + 2, p) * B(p, j); 
        c_30_reg += A(i + 3, p) * B(p, j); 

        // 2nd col
        c_01_reg += A(i, p) * B(p, j + 1); 
        c_11_reg += A(i + 1, p) * B(p, j + 1); 
        c_21_reg += A(i + 2, p) * B(p, j + 1); 
        c_31_reg += A(i + 3, p) * B(p, j + 1); 

        // 3rd col
        c_02_reg += A(i, p) * B(p, j + 2); 
        c_12_reg += A(i + 1, p) * B(p, j + 2); 
        c_22_reg += A(i + 2, p) * B(p, j + 2); 
        c_32_reg += A(i + 3, p) * B(p, j + 2); 

        // 4th col
        c_03_reg += A(i, p) * B(p, j + 3); 
        c_13_reg += A(i + 1, p) * B(p, j + 3); 
        c_23_reg += A(i + 2, p) * B(p, j + 3); 
        c_33_reg += A(i + 3, p) * B(p, j + 3); 
      }
      C(i, j) = c_00_reg;   C(i, j+1) = c_01_reg;   C(i, j+2) = c_02_reg;   C(i, j+3) = c_03_reg;
      C(i+1, j) = c_10_reg;   C(i+1, j+1) = c_11_reg;   C(i+1, j+2) = c_12_reg;   C(i+1, j+3) = c_13_reg;
      C(i+2, j) = c_20_reg;   C(i+2, j+1) = c_21_reg;   C(i+2, j+2) = c_22_reg;   C(i+2, j+3) = c_23_reg;
      C(i+3, j) = c_30_reg;   C(i+3, j+1) = c_31_reg;   C(i+3, j+2) = c_32_reg;   C(i+3, j+3) = c_33_reg;
    }
  }
}

// use register for the 4x4 unrolling + pragma unroll
// no difference with raw `register for the 4x4 unrolling`, aka, MMult_optim4_2
void MMult_optim4_3(float *A, float *B, float *C, int M, int K, int N)
{
  register float  c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg;
  // c_mn_reg means c_(i+m)_(j+n)_reg

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      #pragma unroll
      for (int p = 0; p < K; ++p) {
        // 1st col
        c_00_reg += A(i, p) * B(p, j); 
        c_10_reg += A(i + 1, p) * B(p, j); 
        c_20_reg += A(i + 2, p) * B(p, j); 
        c_30_reg += A(i + 3, p) * B(p, j); 

        // 2nd col
        c_01_reg += A(i, p) * B(p, j + 1); 
        c_11_reg += A(i + 1, p) * B(p, j + 1); 
        c_21_reg += A(i + 2, p) * B(p, j + 1); 
        c_31_reg += A(i + 3, p) * B(p, j + 1); 

        // 3rd col
        c_02_reg += A(i, p) * B(p, j + 2); 
        c_12_reg += A(i + 1, p) * B(p, j + 2); 
        c_22_reg += A(i + 2, p) * B(p, j + 2); 
        c_32_reg += A(i + 3, p) * B(p, j + 2); 

        // 4th col
        c_03_reg += A(i, p) * B(p, j + 3); 
        c_13_reg += A(i + 1, p) * B(p, j + 3); 
        c_23_reg += A(i + 2, p) * B(p, j + 3); 
        c_33_reg += A(i + 3, p) * B(p, j + 3); 
      }
      C(i, j) = c_00_reg;   C(i, j+1) = c_01_reg;   C(i, j+2) = c_02_reg;   C(i, j+3) = c_03_reg;
      C(i+1, j) = c_10_reg;   C(i+1, j+1) = c_11_reg;   C(i+1, j+2) = c_12_reg;   C(i+1, j+3) = c_13_reg;
      C(i+2, j) = c_20_reg;   C(i+2, j+1) = c_21_reg;   C(i+2, j+2) = c_22_reg;   C(i+2, j+3) = c_23_reg;
      C(i+3, j) = c_30_reg;   C(i+3, j+1) = c_31_reg;   C(i+3, j+2) = c_32_reg;   C(i+3, j+3) = c_33_reg;
    }
  }
}



