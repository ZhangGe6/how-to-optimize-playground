#include "params.h"
#include "MMult.h"

// use cache blocking

// suffer from floating-point rounding error problem
// because C_value is added from the results of segments along K dimension, rathen than added one by one in the baseline version
void MMult_optim7_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
    int BLOCKSIZE = 40;
    for (int mBlockStart = 0; mBlockStart < M; mBlockStart += BLOCKSIZE) {
        for (int nBlockStart = 0; nBlockStart < N; nBlockStart += BLOCKSIZE) {
            for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BLOCKSIZE) {
                // printf("mBlockStart %d, nBlockStart %d, kBlockStart %d\n", mBlockStart, nBlockStart, kBlockStart);
                // MMult_base(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);

                // MMult_optim3_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
                // MMult_optim4_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);

                // MMult_optim3_4(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
                // MMult_optim4_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
            }
        }
    }
}

// use blocking based on MMult_optim6_2
// M and N dimension are blocked, this is a base version of MMult_optim7_3
void MMult_optim7_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{

  v2f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v2f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v2f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
    //   float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;
  float *b_p_j0;

  // c_mn_reg means c_(i+m)_(j+n)_reg

  int BLOCKSIZE = 40;
  for (int mBlockStart = 0; mBlockStart < M; mBlockStart += BLOCKSIZE) {
      for (int nBlockStart = 0; nBlockStart < N; nBlockStart += BLOCKSIZE) {

        for (int i = mBlockStart; i < mBlockStart + BLOCKSIZE; i += 4) {
            for (int j = nBlockStart; j < nBlockStart + BLOCKSIZE; j += 4) {
            //   c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
            //   c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
            //   c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
            //   c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;
            c_row_0.reg = _mm_setzero_ps();
            c_row_1.reg = _mm_setzero_ps();
            c_row_2.reg = _mm_setzero_ps();
            c_row_3.reg = _mm_setzero_ps();

            b_p_j0 = &(B(0, j));
            //   b_p_j1 = &(B(0, j + 1));
            //   b_p_j2 = &(B(0, j + 2));
            //   b_p_j3 = &(B(0, j + 3));

            a_i0_p = &(A(i, 0));
            a_i1_p = &(A(i + 1, 0));
            a_i2_p = &(A(i + 2, 0));
            a_i3_p = &(A(i + 3, 0));
            
            for (int p = 0; p < K; ++p) {
                // a_0p_reg = *a_i0_p; 
                // a_1p_reg = *a_i1_p;
                // a_2p_reg = *a_i2_p;
                // a_3p_reg = *a_i3_p;
                a_0p_v.reg = _mm_set_ps1(*a_i0_p);
                a_1p_v.reg = _mm_set_ps1(*a_i1_p);
                a_2p_v.reg = _mm_set_ps1(*a_i2_p);
                a_3p_v.reg = _mm_set_ps1(*a_i3_p);

                // b_p0_reg = *b_p_j0;
                // b_p1_reg = *b_p_j1;
                // b_p2_reg = *b_p_j2;
                // b_p3_reg = *b_p_j3;
                b_p.reg = _mm_load_ps((float *)b_p_j0);

                // c_00_reg += a_0p_reg * b_p0_reg;
                // c_01_reg += a_0p_reg * b_p1_reg;
                // c_02_reg += a_0p_reg * b_p2_reg;
                // c_03_reg += a_0p_reg * b_p3_reg;
                c_row_0.reg += a_0p_v.reg * b_p.reg;

                // c_10_reg += a_1p_reg * b_p0_reg;
                // c_11_reg += a_1p_reg * b_p1_reg;
                // c_12_reg += a_1p_reg * b_p2_reg;
                // c_13_reg += a_1p_reg * b_p3_reg;
                c_row_1.reg += a_1p_v.reg * b_p.reg;

                // c_20_reg += a_2p_reg * b_p0_reg;
                // c_21_reg += a_2p_reg * b_p1_reg;
                // c_22_reg += a_2p_reg * b_p2_reg;
                // c_23_reg += a_2p_reg * b_p3_reg;
                c_row_2.reg += a_2p_v.reg * b_p.reg;

                // c_30_reg += a_3p_reg * b_p0_reg;
                // c_31_reg += a_3p_reg * b_p1_reg;
                // c_32_reg += a_3p_reg * b_p2_reg;
                // c_33_reg += a_3p_reg * b_p3_reg;
                c_row_3.reg += a_3p_v.reg * b_p.reg;

                a_i0_p += 1;
                a_i1_p += 1;
                a_i2_p += 1;
                a_i3_p += 1;

                b_p_j0 += ldb;
                // b_p_j1 += N;
                // b_p_j2 += N;
                // b_p_j3 += N;

            }
            C(i, j) += c_row_0.value[0];   C(i, j+1) += c_row_0.value[1];   C(i, j+2) += c_row_0.value[2];   C(i, j+3) += c_row_0.value[3];
            C(i+1, j) += c_row_1.value[0];   C(i+1, j+1) += c_row_1.value[1];   C(i+1, j+2) += c_row_1.value[2];   C(i+1, j+3) += c_row_1.value[3];
            C(i+2, j) += c_row_2.value[0];   C(i+2, j+1) += c_row_2.value[1];   C(i+2, j+2) += c_row_2.value[2];   C(i+2, j+3) += c_row_2.value[3];
            C(i+3, j) += c_row_3.value[0];   C(i+3, j+1) += c_row_3.value[1];   C(i+3, j+2) += c_row_3.value[2];   C(i+3, j+3) += c_row_3.value[3];
            }
        }
      }
  }
}

// use blocking based on MMult_optim6_2
// block K
// pay attention to the way of summing C value, to avoid floating-point rounding error problem
void MMult_optim7_3(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{

  v2f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v2f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v2f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
    //   float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;
  float *b_p_j0;

  // c_mn_reg means c_(i+m)_(j+n)_reg

  const int BLOCKSIZE = 40;
  #pragma unroll
  for (int mBlockStart = 0; mBlockStart < M; mBlockStart += BLOCKSIZE) {
      #pragma unroll
      for (int nBlockStart = 0; nBlockStart < N; nBlockStart += BLOCKSIZE) {
        #pragma unroll
        for (int i = mBlockStart; i < mBlockStart + BLOCKSIZE; i += 4) {
            #pragma unroll
            for (int j = nBlockStart; j < nBlockStart + BLOCKSIZE; j += 4) {
                //   c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
                //   c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
                //   c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
                //   c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;
                c_row_0.reg = _mm_setzero_ps();
                c_row_1.reg = _mm_setzero_ps();
                c_row_2.reg = _mm_setzero_ps();
                c_row_3.reg = _mm_setzero_ps();

                b_p_j0 = &(B(0, j));
                //   b_p_j1 = &(B(0, j + 1));
                //   b_p_j2 = &(B(0, j + 2));
                //   b_p_j3 = &(B(0, j + 3));

                a_i0_p = &(A(i, 0));
                a_i1_p = &(A(i + 1, 0));
                a_i2_p = &(A(i + 2, 0));
                a_i3_p = &(A(i + 3, 0));
                
                #pragma unroll
                for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BLOCKSIZE) {
                    #pragma unroll
                    for (int p = kBlockStart; p < kBlockStart + BLOCKSIZE; ++p) {
                        // a_0p_reg = *a_i0_p; 
                        // a_1p_reg = *a_i1_p;
                        // a_2p_reg = *a_i2_p;
                        // a_3p_reg = *a_i3_p;
                        a_0p_v.reg = _mm_set_ps1(*a_i0_p);
                        a_1p_v.reg = _mm_set_ps1(*a_i1_p);
                        a_2p_v.reg = _mm_set_ps1(*a_i2_p);
                        a_3p_v.reg = _mm_set_ps1(*a_i3_p);

                        // b_p0_reg = *b_p_j0;
                        // b_p1_reg = *b_p_j1;
                        // b_p2_reg = *b_p_j2;
                        // b_p3_reg = *b_p_j3;
                        b_p.reg = _mm_load_ps((float *)b_p_j0);

                        // c_00_reg += a_0p_reg * b_p0_reg;
                        // c_01_reg += a_0p_reg * b_p1_reg;
                        // c_02_reg += a_0p_reg * b_p2_reg;
                        // c_03_reg += a_0p_reg * b_p3_reg;
                        c_row_0.reg += a_0p_v.reg * b_p.reg;

                        // c_10_reg += a_1p_reg * b_p0_reg;
                        // c_11_reg += a_1p_reg * b_p1_reg;
                        // c_12_reg += a_1p_reg * b_p2_reg;
                        // c_13_reg += a_1p_reg * b_p3_reg;
                        c_row_1.reg += a_1p_v.reg * b_p.reg;

                        // c_20_reg += a_2p_reg * b_p0_reg;
                        // c_21_reg += a_2p_reg * b_p1_reg;
                        // c_22_reg += a_2p_reg * b_p2_reg;
                        // c_23_reg += a_2p_reg * b_p3_reg;
                        c_row_2.reg += a_2p_v.reg * b_p.reg;

                        // c_30_reg += a_3p_reg * b_p0_reg;
                        // c_31_reg += a_3p_reg * b_p1_reg;
                        // c_32_reg += a_3p_reg * b_p2_reg;
                        // c_33_reg += a_3p_reg * b_p3_reg;
                        c_row_3.reg += a_3p_v.reg * b_p.reg;

                        a_i0_p += 1;
                        a_i1_p += 1;
                        a_i2_p += 1;
                        a_i3_p += 1;

                        b_p_j0 += ldb;
                        // b_p_j1 += N;
                        // b_p_j2 += N;
                        // b_p_j3 += N;

                    }
                }

                C(i, j) += c_row_0.value[0];   C(i, j+1) += c_row_0.value[1];   C(i, j+2) += c_row_0.value[2];   C(i, j+3) += c_row_0.value[3];
                C(i+1, j) += c_row_1.value[0];   C(i+1, j+1) += c_row_1.value[1];   C(i+1, j+2) += c_row_1.value[2];   C(i+1, j+3) += c_row_1.value[3];
                C(i+2, j) += c_row_2.value[0];   C(i+2, j+1) += c_row_2.value[1];   C(i+2, j+2) += c_row_2.value[2];   C(i+2, j+3) += c_row_2.value[3];
                C(i+3, j) += c_row_3.value[0];   C(i+3, j+1) += c_row_3.value[1];   C(i+3, j+2) += c_row_3.value[2];   C(i+3, j+3) += c_row_3.value[3];
            }
        }
      }
  }
}

