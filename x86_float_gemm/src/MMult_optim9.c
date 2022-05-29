#include "params.h"
#include "MMult.h"

// packing
// I comprehend the packing sequence by a more premitive code version (like MMult_optim5_1)
// So I know the direction of A (N manner) and B (Z manner), respectively

// Z mode packing
void packB(float *BBlock, float *packedBBlock, int kBlockSize, int nBlockSize, int ldb, int kernel_size) { 
  for (int i = 0; i < nBlockSize; i += 4) {     // colomn wise
    for (int p = 0; p < kBlockSize; ++p) {      // row wise
        float *b_p0 = &BBlock[p * ldb + i];     // &BBlock(p, i);
        for (int j = 0; j < kernel_size; ++j) { // colomn wise
            *packedBBlock++ = *(b_p0 + j);
        }
    }
  }
}

// N mode packing
void packA(float *ABlock, float *packedABlock, int mBlockSize, int kBlockSize, int lda, int kernel_size/*=4*/) {
  for (int i = 0; i < mBlockSize; i += 4) {
    float *a_i_p = &ABlock[i * lda];           // ABlock[i, 0]
    float *a_i1_p = &ABlock[(i + 1) * lda];    // ABlock[i + 1, 0]
    float *a_i2_p = &ABlock[(i + 2) * lda];    // ABlock[i + 2, 0]
    float *a_i3_p = &ABlock[(i + 3) * lda];    // ABlock[i + 3, 0]

    for (int p = 0; p < kBlockSize; ++p) {
      *packedABlock++ = *a_i_p++;
      *packedABlock++ = *a_i1_p++;
      *packedABlock++ = *a_i2_p++;
      *packedABlock++ = *a_i3_p++;
    }

  }
}

// based on MMult_optim6_3
// Note this function is supposed to be used to compute a [sub-block] of C
void MMult_optim_innerBlock_packB(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  v4f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v4f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v4f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  // float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;
  float *b_p_j0;

  // c_mn_reg means c_(i+m)_(j+n)_reg
  float *packedB = (float*) malloc(K * N * sizeof(float));
  // printf("before packing\n");
  // print_matrix(B, K, N, ldb);
  packB(B, packedB, K, N, ldb, /*kernel_size=*/4);
  // printf("packing done\n");
  // print_matrix(packedB, K, 4, 4);

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      // use last C value to initilize registers, to fight against floating-point rounding error problem when cache blocking is used later
      // c_row_0.reg = _mm_set_ps1(C(i, j));
      // c_row_1.reg = _mm_set_ps1(C(i + 1, j));
      // c_row_2.reg = _mm_set_ps1(C(i + 2, j));
      // c_row_3.reg = _mm_set_ps1(C(i + 3, j));
      c_row_0.reg = _mm_load_ps(&C(i, j));
      c_row_1.reg = _mm_load_ps(&C(i + 1, j));
      c_row_2.reg = _mm_load_ps(&C(i + 2, j));
      c_row_3.reg = _mm_load_ps(&C(i + 3, j));

      // b_p_j0 = &(B(0, j));
      b_p_j0 = &(packedB[j * K]);

      a_i0_p = &(A(i, 0));
      a_i1_p = &(A(i + 1, 0));
      a_i2_p = &(A(i + 2, 0));
      a_i3_p = &(A(i + 3, 0));
    
      for (int p = 0; p < K; ++p) {
        a_0p_v.reg = _mm_set_ps1(*a_i0_p);
        a_1p_v.reg = _mm_set_ps1(*a_i1_p);
        a_2p_v.reg = _mm_set_ps1(*a_i2_p);
        a_3p_v.reg = _mm_set_ps1(*a_i3_p);

        b_p.reg = _mm_load_ps((float *)b_p_j0);

        // c_row_0.reg += a_0p_v.reg * b_p.reg;
        c_row_0.reg = _mm_fmadd_ps(a_0p_v.reg, b_p.reg, c_row_0.reg);

        // c_row_1.reg += a_1p_v.reg * b_p.reg;
        c_row_1.reg = _mm_fmadd_ps(a_1p_v.reg, b_p.reg, c_row_1.reg);

        // c_row_2.reg += a_2p_v.reg * b_p.reg;
        c_row_2.reg = _mm_fmadd_ps(a_2p_v.reg, b_p.reg, c_row_2.reg);

        // c_row_3.reg += a_3p_v.reg * b_p.reg;
        c_row_3.reg = _mm_fmadd_ps(a_3p_v.reg, b_p.reg, c_row_3.reg);

        a_i0_p += 1;
        a_i1_p += 1;
        a_i2_p += 1;
        a_i3_p += 1;

        b_p_j0 += 4;

      }
      C(i, j) = c_row_0.value[0];   C(i, j+1) = c_row_0.value[1];   C(i, j+2) = c_row_0.value[2];   C(i, j+3) = c_row_0.value[3];
      C(i+1, j) = c_row_1.value[0];   C(i+1, j+1) = c_row_1.value[1];   C(i+1, j+2) = c_row_1.value[2];   C(i+1, j+3) = c_row_1.value[3];
      C(i+2, j) = c_row_2.value[0];   C(i+2, j+1) = c_row_2.value[1];   C(i+2, j+2) = c_row_2.value[2];   C(i+2, j+3) = c_row_2.value[3];
      C(i+3, j) = c_row_3.value[0];   C(i+3, j+1) = c_row_3.value[1];   C(i+3, j+2) = c_row_3.value[2];   C(i+3, j+3) = c_row_3.value[3];
    }
  }
}

// based on MMult_optim6_3
// Note this function is supposed to be used to compute a [sub-block] of C
void MMult_optim_innerBlock_packA(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  v4f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v4f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v4f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  // float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;
  float *b_p_j0;

  // c_mn_reg means c_(i+m)_(j+n)_reg
  float *packedA = (float*) malloc(M * K * sizeof(float));
  // printf("before packing\n");
  // print_matrix(B, K, N, ldb);
  packA(A, packedA, M, K, lda, /*kernel_size=*/4);

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      // use last C value to initilize registers, to fight against floating-point rounding error problem when cache blocking is used later
      // c_row_0.reg = _mm_set_ps1(C(i, j));
      // c_row_1.reg = _mm_set_ps1(C(i + 1, j));
      // c_row_2.reg = _mm_set_ps1(C(i + 2, j));
      // c_row_3.reg = _mm_set_ps1(C(i + 3, j));
      c_row_0.reg = _mm_load_ps(&C(i, j));
      c_row_1.reg = _mm_load_ps(&C(i + 1, j));
      c_row_2.reg = _mm_load_ps(&C(i + 2, j));
      c_row_3.reg = _mm_load_ps(&C(i + 3, j));

      b_p_j0 = &(B(0, j));

      // a_i0_p = &(A(i, 0));
      // a_i1_p = &(A(i + 1, 0));
      // a_i2_p = &(A(i + 2, 0));
      // a_i3_p = &(A(i + 3, 0));
      a_i0_p = &packedA[i * K];
      a_i1_p = &packedA[i * K + 1];
      a_i2_p = &packedA[i * K + 2];
      a_i3_p = &packedA[i * K + 3];
    
      for (int p = 0; p < K; ++p) {
        a_0p_v.reg = _mm_set_ps1(*a_i0_p);
        a_1p_v.reg = _mm_set_ps1(*a_i1_p);
        a_2p_v.reg = _mm_set_ps1(*a_i2_p);
        a_3p_v.reg = _mm_set_ps1(*a_i3_p);

        b_p.reg = _mm_load_ps((float *)b_p_j0);

        // c_row_0.reg += a_0p_v.reg * b_p.reg;
        c_row_0.reg = _mm_fmadd_ps(a_0p_v.reg, b_p.reg, c_row_0.reg);

        // c_row_1.reg += a_1p_v.reg * b_p.reg;
        c_row_1.reg = _mm_fmadd_ps(a_1p_v.reg, b_p.reg, c_row_1.reg);

        // c_row_2.reg += a_2p_v.reg * b_p.reg;
        c_row_2.reg = _mm_fmadd_ps(a_2p_v.reg, b_p.reg, c_row_2.reg);

        // c_row_3.reg += a_3p_v.reg * b_p.reg;
        c_row_3.reg = _mm_fmadd_ps(a_3p_v.reg, b_p.reg, c_row_3.reg);

        // a_i0_p += 1;
        // a_i1_p += 1;
        // a_i2_p += 1;
        // a_i3_p += 1;

        a_i0_p += 4;
        a_i1_p += 4;
        a_i2_p += 4;
        a_i3_p += 4;

        b_p_j0 += ldb;

      }
      C(i, j) = c_row_0.value[0];   C(i, j+1) = c_row_0.value[1];   C(i, j+2) = c_row_0.value[2];   C(i, j+3) = c_row_0.value[3];
      C(i+1, j) = c_row_1.value[0];   C(i+1, j+1) = c_row_1.value[1];   C(i+1, j+2) = c_row_1.value[2];   C(i+1, j+3) = c_row_1.value[3];
      C(i+2, j) = c_row_2.value[0];   C(i+2, j+1) = c_row_2.value[1];   C(i+2, j+2) = c_row_2.value[2];   C(i+2, j+3) = c_row_2.value[3];
      C(i+3, j) = c_row_3.value[0];   C(i+3, j+1) = c_row_3.value[1];   C(i+3, j+2) = c_row_3.value[2];   C(i+3, j+3) = c_row_3.value[3];
    }
  }
}

void MMult_optim_innerBlock_packAB(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
  v4f_regv c_row_0, c_row_1, c_row_2, c_row_3;
  v4f_regv a_0p_v, a_1p_v, a_2p_v, a_3p_v;
  v4f_regv b_p;

  float *a_i0_p, *a_i1_p, *a_i2_p, *a_i3_p;
  // float *b_p_j0, *b_p_j1, *b_p_j2, *b_p_j3;
  float *b_p_j0;

  // c_mn_reg means c_(i+m)_(j+n)_reg
  float *packedA = (float*) malloc(M * K * sizeof(float));
  float *packedB = (float*) malloc(K * N * sizeof(float));
  // printf("before packing\n");
  // print_matrix(B, K, N, ldb);
  packA(A, packedA, M, K, lda, /*kernel_size=*/4);
  packB(B, packedB, K, N, ldb, /*kernel_size=*/4);

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      // use last C value to initilize registers, to fight against floating-point rounding error problem when cache blocking is used later
      // c_row_0.reg = _mm_set_ps1(C(i, j));
      // c_row_1.reg = _mm_set_ps1(C(i + 1, j));
      // c_row_2.reg = _mm_set_ps1(C(i + 2, j));
      // c_row_3.reg = _mm_set_ps1(C(i + 3, j));
      c_row_0.reg = _mm_load_ps(&C(i, j));
      c_row_1.reg = _mm_load_ps(&C(i + 1, j));
      c_row_2.reg = _mm_load_ps(&C(i + 2, j));
      c_row_3.reg = _mm_load_ps(&C(i + 3, j));

      // b_p_j0 = &(B(0, j));
      b_p_j0 = &(packedB[j * K]);

      // a_i0_p = &(A(i, 0));
      // a_i1_p = &(A(i + 1, 0));
      // a_i2_p = &(A(i + 2, 0));
      // a_i3_p = &(A(i + 3, 0));
      a_i0_p = &packedA[i * K];
      a_i1_p = &packedA[i * K + 1];
      a_i2_p = &packedA[i * K + 2];
      a_i3_p = &packedA[i * K + 3];
    
      for (int p = 0; p < K; ++p) {
        a_0p_v.reg = _mm_set_ps1(*a_i0_p);
        a_1p_v.reg = _mm_set_ps1(*a_i1_p);
        a_2p_v.reg = _mm_set_ps1(*a_i2_p);
        a_3p_v.reg = _mm_set_ps1(*a_i3_p);

        b_p.reg = _mm_load_ps((float *)b_p_j0);

        // c_row_0.reg += a_0p_v.reg * b_p.reg;
        c_row_0.reg = _mm_fmadd_ps(a_0p_v.reg, b_p.reg, c_row_0.reg);

        // c_row_1.reg += a_1p_v.reg * b_p.reg;
        c_row_1.reg = _mm_fmadd_ps(a_1p_v.reg, b_p.reg, c_row_1.reg);

        // c_row_2.reg += a_2p_v.reg * b_p.reg;
        c_row_2.reg = _mm_fmadd_ps(a_2p_v.reg, b_p.reg, c_row_2.reg);

        // c_row_3.reg += a_3p_v.reg * b_p.reg;
        c_row_3.reg = _mm_fmadd_ps(a_3p_v.reg, b_p.reg, c_row_3.reg);

        // a_i0_p += 1;
        // a_i1_p += 1;
        // a_i2_p += 1;
        // a_i3_p += 1;
        a_i0_p += 4;
        a_i1_p += 4;
        a_i2_p += 4;
        a_i3_p += 4;

        // b_p_j0 += ldb;
        b_p_j0 += 4;


      }
      C(i, j) = c_row_0.value[0];   C(i, j+1) = c_row_0.value[1];   C(i, j+2) = c_row_0.value[2];   C(i, j+3) = c_row_0.value[3];
      C(i+1, j) = c_row_1.value[0];   C(i+1, j+1) = c_row_1.value[1];   C(i+1, j+2) = c_row_1.value[2];   C(i+1, j+3) = c_row_1.value[3];
      C(i+2, j) = c_row_2.value[0];   C(i+2, j+1) = c_row_2.value[1];   C(i+2, j+2) = c_row_2.value[2];   C(i+2, j+3) = c_row_2.value[3];
      C(i+3, j) = c_row_3.value[0];   C(i+3, j+1) = c_row_3.value[1];   C(i+3, j+2) = c_row_3.value[2];   C(i+3, j+3) = c_row_3.value[3];
    }
  }
}

void MMult_optim9_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
    int blockSize = 40;
    for (int mBlockStart = 0; mBlockStart < M; mBlockStart += blockSize) {
        for (int nBlockStart = 0; nBlockStart < N; nBlockStart += blockSize) {
            for (int kBlockStart = 0; kBlockStart < K; kBlockStart += blockSize) {
              // printf("mBlockStart %d, nBlockStart %d, kBlockStart %d\n", mBlockStart, nBlockStart, kBlockStart);
              // MMult_optim_innerBlock_packB(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
              // MMult_optim_innerBlock_packA(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
              MMult_optim_innerBlock_packAB(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
            }
        }
    }
}