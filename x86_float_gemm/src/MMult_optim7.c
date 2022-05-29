#include "params.h"
#include "MMult.h"

// use wider [vector] registers (SIMD) __m256
// -mavx or -mavx2 is needed for compiling

// based on MMult_optim6_1. compute 4x8 at a time
// Firstly, rearrange the inner computation
void MMult_optim7_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  register float  c_00_reg, c_01_reg, c_02_reg, c_03_reg, c_04_reg, c_05_reg, c_06_reg, c_07_reg,  
                  c_10_reg, c_11_reg, c_12_reg, c_13_reg, c_14_reg, c_15_reg, c_16_reg, c_17_reg, 
                  c_20_reg, c_21_reg, c_22_reg, c_23_reg, c_24_reg, c_25_reg, c_26_reg, c_27_reg,  
                  c_30_reg, c_31_reg, c_32_reg, c_33_reg, c_34_reg, c_35_reg, c_36_reg, c_37_reg,

                  a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,  // 4 rows
                  b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg, b_p4_reg, b_p5_reg, b_p6_reg, b_p7_reg;

  // c_mn_reg means c_(i+m)_(j+n)_reg
  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3, *b_p_j4, *b_p_j5, *b_p_j6, *b_p_j7;

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 8) {
      // use last C value to initilize registers, to fight against floating-point rounding error problem when cache blocking is used later
      // and C should be initilized to 0 at the very first.
      // c_xx_reg[i:i + 4, j:j + 4]
      c_00_reg = C(i, j);        c_01_reg = C(i, j + 1);        c_02_reg = C(i, j + 2);       c_03_reg = C(i, j + 3);
      c_10_reg = C(i + 1, j);    c_11_reg = C(i + 1, j + 1);    c_12_reg = C(i + 1, j + 2);   c_13_reg = C(i + 1, j + 3);
      c_20_reg = C(i + 2, j);    c_21_reg = C(i + 2, j + 1);    c_22_reg = C(i + 2, j + 2);   c_23_reg = C(i + 2, j + 3);
      c_30_reg = C(i + 3, j);    c_31_reg = C(i + 3, j + 1);    c_32_reg = C(i + 3, j + 2);   c_33_reg = C(i + 3, j + 3);
      // c_xx_reg[i:i + 4, j + 4:j + 8]
      c_04_reg = C(i, j + 4);        c_05_reg = C(i, j + 5);        c_06_reg = C(i, j + 6);       c_07_reg = C(i, j + 7);
      c_14_reg = C(i + 1, j + 4);    c_15_reg = C(i + 1, j + 5);    c_16_reg = C(i + 1, j + 6);   c_17_reg = C(i + 1, j + 7);
      c_24_reg = C(i + 2, j + 4);    c_25_reg = C(i + 2, j + 5);    c_26_reg = C(i + 2, j + 6);   c_27_reg = C(i + 2, j + 7);
      c_34_reg = C(i + 3, j + 4);    c_35_reg = C(i + 3, j + 5);    c_36_reg = C(i + 3, j + 6);   c_37_reg = C(i + 3, j + 7);

      b_p_j0 = &(B(0, j));
      b_p_j1 = &(B(0, j + 1));
      b_p_j2 = &(B(0, j + 2));
      b_p_j3 = &(B(0, j + 3));
      b_p_j4 = &(B(0, j + 4));
      b_p_j5 = &(B(0, j + 5));
      b_p_j6 = &(B(0, j + 6));
      b_p_j7 = &(B(0, j + 7));
  
      a_i0_p = &(A(i, 0));
      a_i1_p = &(A(i + 1, 0));
      a_i2_p = &(A(i + 2, 0));
      a_i3_p = &(A(i + 3, 0));

      for (int p = 0; p < K; ++p) {
        a_0p_reg = *a_i0_p; 
        a_1p_reg = *a_i1_p;
        a_2p_reg = *a_i2_p;
        a_3p_reg = *a_i3_p;

        b_p0_reg = *b_p_j0;
        b_p1_reg = *b_p_j1;
        b_p2_reg = *b_p_j2;
        b_p3_reg = *b_p_j3;
        b_p4_reg = *b_p_j4;
        b_p5_reg = *b_p_j5;
        b_p6_reg = *b_p_j6;
        b_p7_reg = *b_p_j7;

        // 1st row
        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;
        c_02_reg += a_0p_reg * b_p2_reg;
        c_03_reg += a_0p_reg * b_p3_reg;
        c_04_reg += a_0p_reg * b_p4_reg;
        c_05_reg += a_0p_reg * b_p5_reg;
        c_06_reg += a_0p_reg * b_p6_reg;
        c_07_reg += a_0p_reg * b_p7_reg;

        // 2nd row
        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;
        c_12_reg += a_1p_reg * b_p2_reg;
        c_13_reg += a_1p_reg * b_p3_reg;
        c_14_reg += a_1p_reg * b_p4_reg;
        c_15_reg += a_1p_reg * b_p5_reg;
        c_16_reg += a_1p_reg * b_p6_reg;
        c_17_reg += a_1p_reg * b_p7_reg;

        // 3rd row
        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;
        c_22_reg += a_2p_reg * b_p2_reg;
        c_23_reg += a_2p_reg * b_p3_reg;
        c_24_reg += a_2p_reg * b_p4_reg;
        c_25_reg += a_2p_reg * b_p5_reg;
        c_26_reg += a_2p_reg * b_p6_reg;
        c_27_reg += a_2p_reg * b_p7_reg;

        // 4th row
        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;
        c_32_reg += a_3p_reg * b_p2_reg;
        c_33_reg += a_3p_reg * b_p3_reg;
        c_34_reg += a_3p_reg * b_p4_reg;
        c_35_reg += a_3p_reg * b_p5_reg;
        c_36_reg += a_3p_reg * b_p6_reg;
        c_37_reg += a_3p_reg * b_p7_reg;

        // update pointers
        a_i0_p += 1;
        a_i1_p += 1;
        a_i2_p += 1;
        a_i3_p += 1;

        b_p_j0 += ldb;
        b_p_j1 += ldb;
        b_p_j2 += ldb;
        b_p_j3 += ldb;
        b_p_j4 += ldb;
        b_p_j5 += ldb;
        b_p_j6 += ldb;
        b_p_j7 += ldb;
      }
      // c_xx_reg[i:i + 4, j:j + 4]
      C(i, j) = c_00_reg;   C(i, j+1) = c_01_reg;   C(i, j+2) = c_02_reg;   C(i, j+3) = c_03_reg;
      C(i+1, j) = c_10_reg;   C(i+1, j+1) = c_11_reg;   C(i+1, j+2) = c_12_reg;   C(i+1, j+3) = c_13_reg;
      C(i+2, j) = c_20_reg;   C(i+2, j+1) = c_21_reg;   C(i+2, j+2) = c_22_reg;   C(i+2, j+3) = c_23_reg;
      C(i+3, j) = c_30_reg;   C(i+3, j+1) = c_31_reg;   C(i+3, j+2) = c_32_reg;   C(i+3, j+3) = c_33_reg;
      // c_xx_reg[i:i + 4, j + 4:j + 8]
      C(i, j+4) = c_04_reg;   C(i, j+5) = c_05_reg;   C(i, j+6) = c_06_reg;   C(i, j+7) = c_07_reg;
      C(i+1, j+4) = c_14_reg;   C(i+1, j+5) = c_15_reg;   C(i+1, j+6) = c_16_reg;   C(i+1, j+7) = c_17_reg;
      C(i+2, j+4) = c_24_reg;   C(i+2, j+5) = c_25_reg;   C(i+2, j+6) = c_26_reg;   C(i+2, j+7) = c_27_reg;
      C(i+3, j+4) = c_34_reg;   C(i+3, j+5) = c_35_reg;   C(i+3, j+6) = c_36_reg;   C(i+3, j+7) = c_37_reg;

    }
  }
}

// Then use the vector registers to combine the computations
// larger boost (14->30 GFLOPs)
void MMult_optim7_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  // register float  c_00_reg, c_01_reg, c_02_reg, c_03_reg, c_04_reg, c_05_reg, c_06_reg, c_07_reg,  
  //                 c_10_reg, c_11_reg, c_12_reg, c_13_reg, c_14_reg, c_15_reg, c_16_reg, c_17_reg, 
  //                 c_20_reg, c_21_reg, c_22_reg, c_23_reg, c_24_reg, c_25_reg, c_26_reg, c_27_reg,  
  //                 c_30_reg, c_31_reg, c_32_reg, c_33_reg, c_34_reg, c_35_reg, c_36_reg, c_37_reg,

  //                 a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,  // 4 rows
  //                 b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg, b_p4_reg, b_p5_reg, b_p6_reg, b_p7_reg;

  v8f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v8f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v8f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  // float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3, *b_p_j4, *b_p_j5, *b_p_j6, *b_p_j7;
  float *b_p_j0;

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 8) {

      // This leads to segmentation fault. May caused by memory misalignment
      // c_row_0.reg = _mm256_load_ps(&C(i, j));
      // c_row_1.reg = _mm256_load_ps(&C(i + 1, j));
      // c_row_2.reg = _mm256_load_ps(&C(i + 2, j));
      // c_row_3.reg = _mm256_load_ps(&C(i + 3, j));
  
      c_row_0.reg = _mm256_loadu_ps(&C(i, j));
      c_row_1.reg = _mm256_loadu_ps(&C(i + 1, j));
      c_row_2.reg = _mm256_loadu_ps(&C(i + 2, j));
      c_row_3.reg = _mm256_loadu_ps(&C(i + 3, j));
      // printf("run to here\n");
      
      b_p_j0 = &(B(0, j));
      // b_p_j1 = &(B(0, j + 1));
      // b_p_j2 = &(B(0, j + 2));
      // b_p_j3 = &(B(0, j + 3));
      // b_p_j4 = &(B(0, j + 4));
      // b_p_j5 = &(B(0, j + 5));
      // b_p_j6 = &(B(0, j + 6));
      // b_p_j7 = &(B(0, j + 7));
  
      a_i0_p = &(A(i, 0));
      a_i1_p = &(A(i + 1, 0));
      a_i2_p = &(A(i + 2, 0));
      a_i3_p = &(A(i + 3, 0));

      for (int p = 0; p < K; ++p) {
        // a_0p_reg = *a_i0_p; 
        // a_1p_reg = *a_i1_p;
        // a_2p_reg = *a_i2_p;
        // a_3p_reg = *a_i3_p;
        a_0p_v.reg = _mm256_set1_ps(*a_i0_p);
        a_1p_v.reg = _mm256_set1_ps(*a_i1_p);
        a_2p_v.reg = _mm256_set1_ps(*a_i2_p);
        a_3p_v.reg = _mm256_set1_ps(*a_i3_p);

        // b_p0_reg = *b_p_j0;
        // b_p1_reg = *b_p_j1;
        // b_p2_reg = *b_p_j2;
        // b_p3_reg = *b_p_j3;
        // b_p4_reg = *b_p_j4;
        // b_p5_reg = *b_p_j5;
        // b_p6_reg = *b_p_j6;
        // b_p7_reg = *b_p_j7;
        b_p.reg = _mm256_loadu_ps((float *)b_p_j0);

        // 1st row
        // c_00_reg += a_0p_reg * b_p0_reg;
        // c_01_reg += a_0p_reg * b_p1_reg;
        // c_02_reg += a_0p_reg * b_p2_reg;
        // c_03_reg += a_0p_reg * b_p3_reg;
        // c_04_reg += a_0p_reg * b_p4_reg;
        // c_05_reg += a_0p_reg * b_p5_reg;
        // c_06_reg += a_0p_reg * b_p6_reg;
        // c_07_reg += a_0p_reg * b_p7_reg;
        c_row_0.reg += a_0p_v.reg * b_p.reg;

        // 2nd row
        // c_10_reg += a_1p_reg * b_p0_reg;
        // c_11_reg += a_1p_reg * b_p1_reg;
        // c_12_reg += a_1p_reg * b_p2_reg;
        // c_13_reg += a_1p_reg * b_p3_reg;
        // c_14_reg += a_1p_reg * b_p4_reg;
        // c_15_reg += a_1p_reg * b_p5_reg;
        // c_16_reg += a_1p_reg * b_p6_reg;
        // c_17_reg += a_1p_reg * b_p7_reg;
        c_row_1.reg += a_1p_v.reg * b_p.reg;


        // 3rd row
        // c_20_reg += a_2p_reg * b_p0_reg;
        // c_21_reg += a_2p_reg * b_p1_reg;
        // c_22_reg += a_2p_reg * b_p2_reg;
        // c_23_reg += a_2p_reg * b_p3_reg;
        // c_24_reg += a_2p_reg * b_p4_reg;
        // c_25_reg += a_2p_reg * b_p5_reg;
        // c_26_reg += a_2p_reg * b_p6_reg;
        // c_27_reg += a_2p_reg * b_p7_reg;
        c_row_2.reg += a_2p_v.reg * b_p.reg;

        // 4th row
        // c_30_reg += a_3p_reg * b_p0_reg;
        // c_31_reg += a_3p_reg * b_p1_reg;
        // c_32_reg += a_3p_reg * b_p2_reg;
        // c_33_reg += a_3p_reg * b_p3_reg;
        // c_34_reg += a_3p_reg * b_p4_reg;
        // c_35_reg += a_3p_reg * b_p5_reg;
        // c_36_reg += a_3p_reg * b_p6_reg;
        // c_37_reg += a_3p_reg * b_p7_reg;
        c_row_3.reg += a_3p_v.reg * b_p.reg;

        // update pointers
        a_i0_p += 1;
        a_i1_p += 1;
        a_i2_p += 1;
        a_i3_p += 1;

        b_p_j0 += ldb;
        // b_p_j1 += ldb;
        // b_p_j2 += ldb;
        // b_p_j3 += ldb;
        // b_p_j4 += ldb;
        // b_p_j5 += ldb;
        // b_p_j6 += ldb;
        // b_p_j7 += ldb;
      }
      // c_xx_reg[i:i + 4, j:j + 4]
      C(i, j) = c_row_0.value[0];   C(i, j+1) = c_row_0.value[1];   C(i, j+2) = c_row_0.value[2];   C(i, j+3) = c_row_0.value[3];
      C(i+1, j) = c_row_1.value[0];   C(i+1, j+1) = c_row_1.value[1];   C(i+1, j+2) = c_row_1.value[2];   C(i+1, j+3) = c_row_1.value[3];
      C(i+2, j) = c_row_2.value[0];   C(i+2, j+1) = c_row_2.value[1];   C(i+2, j+2) = c_row_2.value[2];   C(i+2, j+3) = c_row_2.value[3];
      C(i+3, j) = c_row_3.value[0];   C(i+3, j+1) = c_row_3.value[1];   C(i+3, j+2) = c_row_3.value[2];   C(i+3, j+3) = c_row_3.value[3];
      // c_xx_reg[i:i + 4, j + 4:j + 8]
      C(i, j+4) = c_row_0.value[4];   C(i, j+5) = c_row_0.value[5];   C(i, j+6) = c_row_0.value[6];   C(i, j+7) = c_row_0.value[7];
      C(i+1, j+4) = c_row_1.value[4];   C(i+1, j+5) = c_row_1.value[5];   C(i+1, j+6) = c_row_1.value[6];   C(i+1, j+7) = c_row_1.value[7];
      C(i+2, j+4) = c_row_2.value[4];   C(i+2, j+5) = c_row_2.value[5];   C(i+2, j+6) = c_row_2.value[6];   C(i+2, j+7) = c_row_2.value[7];
      C(i+3, j+4) = c_row_3.value[4];   C(i+3, j+5) = c_row_3.value[5];   C(i+3, j+6) = c_row_3.value[6];   C(i+3, j+7) = c_row_3.value[7];
    }
  }
}

// use Fuse Multiply and Add (FMA) instruction
// -mfma flag is needed for compiling
// larger boost (30->40 GFLOPs)
void MMult_optim7_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  v8f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v8f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v8f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  // float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3, *b_p_j4, *b_p_j5, *b_p_j6, *b_p_j7;
  float *b_p_j0;

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 8) {

      c_row_0.reg = _mm256_loadu_ps(&C(i, j));
      c_row_1.reg = _mm256_loadu_ps(&C(i + 1, j));
      c_row_2.reg = _mm256_loadu_ps(&C(i + 2, j));
      c_row_3.reg = _mm256_loadu_ps(&C(i + 3, j));
      // printf("run to here\n");
      
      b_p_j0 = &(B(0, j));

      a_i0_p = &(A(i, 0));
      a_i1_p = &(A(i + 1, 0));
      a_i2_p = &(A(i + 2, 0));
      a_i3_p = &(A(i + 3, 0));

      for (int p = 0; p < K; ++p) {

        a_0p_v.reg = _mm256_set1_ps(*a_i0_p);
        a_1p_v.reg = _mm256_set1_ps(*a_i1_p);
        a_2p_v.reg = _mm256_set1_ps(*a_i2_p);
        a_3p_v.reg = _mm256_set1_ps(*a_i3_p);

        b_p.reg = _mm256_loadu_ps((float *)b_p_j0);

        // 1st row
        // c_row_0.reg += a_0p_v.reg * b_p.reg;
        c_row_0.reg = _mm256_fmadd_ps(a_0p_v.reg, b_p.reg, c_row_0.reg);

        // 2nd row
        // c_row_1.reg += a_1p_v.reg * b_p.reg;
        c_row_1.reg = _mm256_fmadd_ps(a_1p_v.reg, b_p.reg, c_row_1.reg);

        // 3rd row
        // c_row_2.reg += a_2p_v.reg * b_p.reg;
        c_row_2.reg = _mm256_fmadd_ps(a_2p_v.reg, b_p.reg, c_row_2.reg);

        // 4th row
        // c_row_3.reg += a_3p_v.reg * b_p.reg;
        c_row_3.reg = _mm256_fmadd_ps(a_3p_v.reg, b_p.reg, c_row_3.reg);

        // update pointers
        a_i0_p += 1;
        a_i1_p += 1;
        a_i2_p += 1;
        a_i3_p += 1;

        b_p_j0 += ldb;
      }
      // c_xx_reg[i:i + 4, j:j + 4]
      C(i, j) = c_row_0.value[0];   C(i, j+1) = c_row_0.value[1];   C(i, j+2) = c_row_0.value[2];   C(i, j+3) = c_row_0.value[3];
      C(i+1, j) = c_row_1.value[0];   C(i+1, j+1) = c_row_1.value[1];   C(i+1, j+2) = c_row_1.value[2];   C(i+1, j+3) = c_row_1.value[3];
      C(i+2, j) = c_row_2.value[0];   C(i+2, j+1) = c_row_2.value[1];   C(i+2, j+2) = c_row_2.value[2];   C(i+2, j+3) = c_row_2.value[3];
      C(i+3, j) = c_row_3.value[0];   C(i+3, j+1) = c_row_3.value[1];   C(i+3, j+2) = c_row_3.value[2];   C(i+3, j+3) = c_row_3.value[3];
      // c_xx_reg[i:i + 4, j + 4:j + 8]
      C(i, j+4) = c_row_0.value[4];   C(i, j+5) = c_row_0.value[5];   C(i, j+6) = c_row_0.value[6];   C(i, j+7) = c_row_0.value[7];
      C(i+1, j+4) = c_row_1.value[4];   C(i+1, j+5) = c_row_1.value[5];   C(i+1, j+6) = c_row_1.value[6];   C(i+1, j+7) = c_row_1.value[7];
      C(i+2, j+4) = c_row_2.value[4];   C(i+2, j+5) = c_row_2.value[5];   C(i+2, j+6) = c_row_2.value[6];   C(i+2, j+7) = c_row_2.value[7];
      C(i+3, j+4) = c_row_3.value[4];   C(i+3, j+5) = c_row_3.value[5];   C(i+3, j+6) = c_row_3.value[6];   C(i+3, j+7) = c_row_3.value[7];
    }
  }
}