#include "params.h"
#include "MMult.h"
#include "utils.h"

// https://github.com/flame/how-to-optimize-gemm/wiki#computing-a-4-x-4-block-of-c-at-a-time
// gonna to use vector registers

// https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_10.c
// use vector registers
// a large boost compared with MMult_optim6_7
// the comments are kept for easier comparison
void MMult_optim7_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  // register double c_00_reg, c_01_reg, c_02_reg, c_03_reg,  
  //                 c_10_reg, c_11_reg, c_12_reg, c_13_reg,  
  //                 c_20_reg, c_21_reg, c_22_reg, c_23_reg,  
  //                 c_30_reg, c_31_reg, c_32_reg, c_33_reg,
  //                 a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
  //                 b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
  // // c_mn_reg means c_(i+m)_(j+n)_reg

  v2d_regv c_00_01, c_10_11, c_20_21, c_30_31,
           c_02_03, c_12_13, c_22_23, c_32_33;
  v2d_regv b_p0_p1, b_p2_p3;
  v2d_regv a_0p, a_1p, a_2p, a_3p;  // the single value will be duplicated to be a vector

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
      // c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
      // c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
      // c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
      // c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

      // https://www.univ-orleans.fr/lifo/Members/Sylvain.Jubertie/doc/SIMD/html/group__setops.html
      // Returns a vector of 2 DPFP values set to 0.0.
      c_00_01.reg = _mm_setzero_pd();
      c_10_11.reg = _mm_setzero_pd();
      c_20_21.reg = _mm_setzero_pd();
      c_30_31.reg = _mm_setzero_pd();
      c_02_03.reg = _mm_setzero_pd();
      c_12_13.reg = _mm_setzero_pd();
      c_22_23.reg = _mm_setzero_pd();
      c_32_33.reg = _mm_setzero_pd();

      b_p0_pntr = &B(0, j);
      // b_p1_pntr = &B(0, j + 1);
      b_p2_pntr = &B(0, j + 2); 
      // b_p3_pntr = &B(0, j + 3);

      for (int p = 0; p < k; ++p){
        // a_0p_reg = A(i, p); 
        // a_1p_reg = A(i + 1, p);
        // a_2p_reg = A(i + 2, p);
        // a_3p_reg = A(i + 3, p);

        // https://blog.csdn.net/fengbingchun/article/details/21322849
        // extern __m128d _mm_loaddup_pd(double const * dp);  => r0=r1=dp[0]
        a_0p.reg = _mm_loaddup_pd((double *) &A(i, p));
        a_1p.reg = _mm_loaddup_pd((double *) &A(i + 1, p));
        a_2p.reg = _mm_loaddup_pd((double *) &A(i + 2, p));
        a_3p.reg = _mm_loaddup_pd((double *) &A(i + 3, p));

        // b_p0_reg = *b_p0_pntr;
        // b_p1_reg = *b_p1_pntr;
        // b_p2_reg = *b_p2_pntr;
        // b_p3_reg = *b_p3_pntr;

        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_pd&ig_expand=4261
        // __m128d _mm_load_pd (double const* mem_addr)
        // Load 128-bits (composed of 2 packed double-precision (64-bit) floating-point elements) from memory into dst.
        b_p0_p1.reg = _mm_load_pd((double *) b_p0_pntr);
        b_p2_p3.reg = _mm_load_pd((double *) b_p2_pntr);

        /* First colomns and second colomns */
        // c_00_reg += a_0p_reg * b_p0_reg;
        // c_01_reg += a_0p_reg * b_p1_reg;
        c_00_01.reg += a_0p.reg * b_p0_p1.reg;

        // c_10_reg += a_1p_reg * b_p0_reg;
        // c_11_reg += a_1p_reg * b_p1_reg;
        c_10_11.reg += a_1p.reg * b_p0_p1.reg;

        // c_20_reg += a_2p_reg * b_p0_reg;
        // c_21_reg += a_2p_reg * b_p1_reg;
        c_20_21.reg += a_2p.reg * b_p0_p1.reg;

        // c_30_reg += a_3p_reg * b_p0_reg;
        // c_31_reg += a_3p_reg * b_p1_reg;
        c_30_31.reg += a_3p.reg * b_p0_p1.reg;

        /* Third colomns fourth colomns */
        // c_02_reg += a_0p_reg * b_p2_reg;
        // c_03_reg += a_0p_reg * b_p3_reg;
        c_02_03.reg += a_0p.reg * b_p2_p3.reg;

        // c_12_reg += a_1p_reg * b_p2_reg;
        // c_13_reg += a_1p_reg * b_p3_reg;
        c_12_13.reg += a_1p.reg * b_p2_p3.reg;

        // c_22_reg += a_2p_reg * b_p2_reg;
        // c_23_reg += a_2p_reg * b_p3_reg;
        c_22_23.reg += a_2p.reg * b_p2_p3.reg;

        // c_32_reg += a_3p_reg * b_p2_reg;
        // c_33_reg += a_3p_reg * b_p3_reg;
        c_32_33.reg += a_3p.reg * b_p2_p3.reg;

        // update b_px_pntr
        b_p0_pntr += ldb;
        b_p1_pntr += ldb;
        b_p2_pntr += ldb; 
        b_p3_pntr += ldb;
      }
      // C(i, j) += c_00_reg;   C(i, j+1) += c_01_reg;   C(i, j+2) += c_02_reg;   C(i, j+3) += c_03_reg;
      // C(i+1, j) += c_10_reg;   C(i+1, j+1) += c_11_reg;   C(i+1, j+2) += c_12_reg;   C(i+1, j+3) += c_13_reg;
      // C(i+2, j) += c_20_reg;   C(i+2, j+1) += c_21_reg;   C(i+2, j+2) += c_22_reg;   C(i+2, j+3) += c_23_reg;
      // C(i+3, j) += c_30_reg;   C(i+3, j+1) += c_31_reg;   C(i+3, j+2) += c_32_reg;   C(i+3, j+3) += c_33_reg;

      C(i, j) += c_00_01.value[0];     C(i, j+1) += c_00_01.value[1];     C(i, j+2) += c_02_03.value[0];     C(i, j+3) += c_02_03.value[1];
      C(i+1, j) += c_10_11.value[0];   C(i+1, j+1) += c_10_11.value[1];   C(i+1, j+2) += c_12_13.value[0];   C(i+1, j+3) += c_12_13.value[1];
      C(i+2, j) += c_20_21.value[0];   C(i+2, j+1) += c_20_21.value[1];   C(i+2, j+2) += c_22_23.value[0];   C(i+2, j+3) += c_22_23.value[1];
      C(i+3, j) += c_30_31.value[0];   C(i+3, j+1) += c_30_31.value[1];   C(i+3, j+2) += c_32_33.value[0];   C(i+3, j+3) += c_32_33.value[1];
    }
  }  
}
// MMult_optim7_1 works! cheers and keep going on!

