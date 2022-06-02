#include "params.h"
#include "MMult.h"

// use pointer to reduce indexing overhead, as suggested in https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_7
// So what is the reason that pointer can reduce indexing overhead? I GUESS that by using pointer, the calculation of index is simplified (pure addition and no muplication)

// even slightly slower
void MMult_optim5_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  register float  c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg,

                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

  // c_mn_reg means c_(i+m)_(j+n)_reg
  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;


  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      // use last C value to initilize registers, to fight against floating-point rounding error problem when cache blocking is used later
      // and C should be initilized to 0 at the very first.
      c_00_reg = C(i, j);        c_01_reg = C(i, j + 1);        c_02_reg = C(i, j + 2);       c_03_reg = C(i, j + 3);
      c_10_reg = C(i + 1, j);    c_11_reg = C(i + 1, j + 1);    c_12_reg = C(i + 1, j + 2);   c_13_reg = C(i + 1, j + 3);
      c_20_reg = C(i + 2, j);    c_21_reg = C(i + 2, j + 1);    c_22_reg = C(i + 2, j + 2);   c_23_reg = C(i + 2, j + 3);
      c_30_reg = C(i + 3, j);    c_31_reg = C(i + 3, j + 1);    c_32_reg = C(i + 3, j + 2);   c_33_reg = C(i + 3, j + 3);

      b_p_j0 = &(B(0, j));
      b_p_j1 = &(B(0, j + 1));
      b_p_j2 = &(B(0, j + 2));
      b_p_j3 = &(B(0, j + 3));
  
      a_i0_p = &(A(i, 0));
      a_i1_p = &(A(i + 1, 0));
      a_i2_p = &(A(i + 2, 0));
      a_i3_p = &(A(i + 3, 0));

      for (int p = 0; p < K; ++p) {
        // a_0p_reg = A(i, p); 
        // a_1p_reg = A(i + 1, p);
        // a_2p_reg = A(i + 2, p);
        // a_3p_reg = A(i + 3, p);
        a_0p_reg = *a_i0_p; 
        a_1p_reg = *a_i1_p;
        a_2p_reg = *a_i2_p;
        a_3p_reg = *a_i3_p;

        // b_p0_reg = B(p, j);
        // b_p1_reg = B(p, j + 1);
        // b_p2_reg = B(p, j + 2);
        // b_p3_reg = B(p, j + 3);
        b_p0_reg = *b_p_j0;
        b_p1_reg = *b_p_j1;
        b_p2_reg = *b_p_j2;
        b_p3_reg = *b_p_j3;

        // 1st col
        c_00_reg += a_0p_reg * b_p0_reg; 
        c_10_reg += a_1p_reg * b_p0_reg; 
        c_20_reg += a_2p_reg * b_p0_reg; 
        c_30_reg += a_3p_reg * b_p0_reg; 

        // 2nd col
        c_01_reg += a_0p_reg * b_p1_reg; 
        c_11_reg += a_1p_reg * b_p1_reg; 
        c_21_reg += a_2p_reg * b_p1_reg; 
        c_31_reg += a_3p_reg * b_p1_reg; 

        // 3rd col
        c_02_reg += a_0p_reg * b_p2_reg; 
        c_12_reg += a_1p_reg * b_p2_reg; 
        c_22_reg += a_2p_reg * b_p2_reg; 
        c_32_reg += a_3p_reg * b_p2_reg; 

        // 4th col
        c_03_reg += a_0p_reg * b_p3_reg; 
        c_13_reg += a_1p_reg * b_p3_reg; 
        c_23_reg += a_2p_reg * b_p3_reg; 
        c_33_reg += a_3p_reg * b_p3_reg; 

        a_i0_p += 1;
        a_i1_p += 1;
        a_i2_p += 1;
        a_i3_p += 1;

        b_p_j0 += ldb;
        b_p_j1 += ldb;
        b_p_j2 += ldb;
        b_p_j3 += ldb;
      }
      C(i, j) = c_00_reg;   C(i, j+1) = c_01_reg;   C(i, j+2) = c_02_reg;   C(i, j+3) = c_03_reg;
      C(i+1, j) = c_10_reg;   C(i+1, j+1) = c_11_reg;   C(i+1, j+2) = c_12_reg;   C(i+1, j+3) = c_13_reg;
      C(i+2, j) = c_20_reg;   C(i+2, j+1) = c_21_reg;   C(i+2, j+2) = c_22_reg;   C(i+2, j+3) = c_23_reg;
      C(i+3, j) = c_30_reg;   C(i+3, j+1) = c_31_reg;   C(i+3, j+2) = c_32_reg;   C(i+3, j+3) = c_33_reg;
    }
  }
}

