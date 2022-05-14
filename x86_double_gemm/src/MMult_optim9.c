#include "params.h"
#include "MMult.h"
#include "utils.h"

// Packing into contiguous memory
// Not exactly the same with:
// https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_12.c

// How it works
// 1. https://chhzh123.github.io/blogs/2020-03-20-tvm-gemm/#%E6%95%B0%E7%BB%84%E6%89%93%E5%8C%85packing


// TODO: why pack need Z or N style? https://zhuanlan.zhihu.com/p/69700540
// I think it is accroding to cailculation order in the `p` for
// Note that we only pack the right-now-computing A sub-block (4 * kc) per calling
// packing A in `N` mode
void packA(int kBlock, double *A, double *packedA, int lda, int kernel_size) {  // A here is one block of full A 
    for (int p = 0; p < kBlock; ++p) {
      double *a_0p = &A(0, p);
      for (int i = 0; i < kernel_size; ++i) {
        *(packedA + p * kernel_size + i) = *(a_0p + i * p);
      }
    }
}

// Note that we only pack the right-now-computing B sub-block (kc * 4) per calling
// packing B in `Z` mode
void packB(int kBlock, int nBlock, double *B, double *packedB, int ldb, int kernel_size) {  // B here is one block of full B 
    for (int p = 0; p < kBlock; ++p) {
        double *b_p0 = &B(p, 0);
        for (int j = 0; j < kernel_size; ++j) {
            *(packedB + p * kernel_size + j) = *(b_p0 + j);
        }
    }
}

// PackA based on MMult_optim7_1
void MMult_optim7_1_packA(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  // printf("size %d, %d, %d\n", m, k, n);
  v2d_regv c_00_01, c_10_11, c_20_21, c_30_31,
           c_02_03, c_12_13, c_22_23, c_32_33;
  v2d_regv b_p0_p1, b_p2_p3;
  v2d_regv a_0p, a_1p, a_2p, a_3p;  // the single value will be duplicated to be a vector

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  double *packedA = (double*) malloc(m * k * sizeof(double));
  double *packedA_ptr = packedA;
  // This type of packing is very time-costful
  // for (int i = 0; i < m; i += 4) {
  //   for (int p = 0; p < k; ++p) {
  //     *packedA_ptr = A(i, p);   // ATTENTION!: not A[i, p]
  //     *(packedA_ptr + 1) = A(i + 1, p);
  //     *(packedA_ptr + 2) = A(i + 2, p);
  //     *(packedA_ptr + 3) = A(i + 3, p);

  //     packedA_ptr += 4;
  //   }
  // }

  // No speedup
  double *a_ip, *a_i1_p, *a_i2_p, *a_i3_p;
  for (int i = 0; i < m; i += 4) {
    a_ip = &A(i, 0);        // ATTENTION!: not A[i, p]
    a_i1_p = &A(i + 1, 0), 
    a_i2_p = &A(i + 2, 0), 
    a_i3_p = &A(i + 3, 0);

    for (int p = 0; p < k; ++p) {
      *(packedA_ptr++) = *(a_ip++);
      *(packedA_ptr++) = *(a_i1_p++);
      *(packedA_ptr++) = *(a_i2_p++);
      *(packedA_ptr++) = *(a_i3_p++);
    }
  }



  for (int i = 0; i < m; i += 4){
    for (int j = 0; j < n; j += 4){
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

      // avoid to repeat packing A
      // if (j == 0) {
      //   packA(k, &A(i, 0), &packedA[i * k], lda, 4);
      // // packA(k, &A(i, 0), packedA, lda, 4);
      // }
      // printf("packed done, packed sample %f\n", packedA[0]);
      double *packed_a_ptr = &packedA[i * k];  // A(i 0)

      for (int p = 0; p < k; ++p){
        // a_0p.reg = _mm_loaddup_pd((double *) &A(i, p));
        // a_1p.reg = _mm_loaddup_pd((double *) &A(i + 1, p));
        // a_2p.reg = _mm_loaddup_pd((double *) &A(i + 2, p));
        // a_3p.reg = _mm_loaddup_pd((double *) &A(i + 3, p));

        // printf("A(i, p) %f,  packedA val %f\n", A(i, p), *packed_a_i0);

        a_0p.reg = _mm_loaddup_pd((double *) packed_a_ptr);  
        a_1p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 1));
        a_2p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 2));
        a_3p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 3));
        packed_a_ptr += 4;

        b_p0_p1.reg = _mm_load_pd((double *) b_p0_pntr);
        b_p2_p3.reg = _mm_load_pd((double *) b_p2_pntr);

        /* First colomns and second colomns */
        c_00_01.reg += a_0p.reg * b_p0_p1.reg;
        c_10_11.reg += a_1p.reg * b_p0_p1.reg;

        c_20_21.reg += a_2p.reg * b_p0_p1.reg;
        c_30_31.reg += a_3p.reg * b_p0_p1.reg;

        /* Third colomns fourth colomns */
        c_02_03.reg += a_0p.reg * b_p2_p3.reg;
        c_12_13.reg += a_1p.reg * b_p2_p3.reg;

        c_22_23.reg += a_2p.reg * b_p2_p3.reg;
        c_32_33.reg += a_3p.reg * b_p2_p3.reg;

        // update b_px_pntr
        b_p0_pntr += ldb;
        b_p1_pntr += ldb;
        b_p2_pntr += ldb; 
        b_p3_pntr += ldb;
        // printf("p%d\n", p);

      }
      // printf("cece %f", 1000);
      

      C(i, j) += c_00_01.value[0];     C(i, j+1) += c_00_01.value[1];     C(i, j+2) += c_02_03.value[0];     C(i, j+3) += c_02_03.value[1];
      C(i+1, j) += c_10_11.value[0];   C(i+1, j+1) += c_10_11.value[1];   C(i+1, j+2) += c_12_13.value[0];   C(i+1, j+3) += c_12_13.value[1];
      C(i+2, j) += c_20_21.value[0];   C(i+2, j+1) += c_20_21.value[1];   C(i+2, j+2) += c_22_23.value[0];   C(i+2, j+3) += c_22_23.value[1];
      C(i+3, j) += c_30_31.value[0];   C(i+3, j+1) += c_30_31.value[1];   C(i+3, j+2) += c_32_33.value[0];   C(i+3, j+3) += c_32_33.value[1];
    }
  }  
}

// PackB based on MMult_optim7_1
void MMult_optim7_1_packB(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  // printf("size %d, %d, %d\n", m, k, n);
  v2d_regv c_00_01, c_10_11, c_20_21, c_30_31,
           c_02_03, c_12_13, c_22_23, c_32_33;
  v2d_regv b_p0_p1, b_p2_p3;
  v2d_regv a_0p, a_1p, a_2p, a_3p;  // the single value will be duplicated to be a vector

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  // double *packedB = (double*) malloc(k * n * sizeof(double));
  // double *packedB_ptr = packedB;
  // for (int j = 0; j < n; j += 4) {
  //   for (int p = 0; p < k; ++p) {
  //     *packedB_ptr = B(p, j);
  //     *(packedB_ptr + 1) = B(p, j + 1);
  //     *(packedB_ptr + 2) = B(p, j + 2);
  //     *(packedB_ptr + 3) = B(p, j + 3);

  //     packedB_ptr += 4;
  //   }
  // }

  // No speedup too
  double *packedB = (double*) malloc(k * n * sizeof(double));
  double *packedB_ptr = packedB;
  double *b_p_j;
  for (int j = 0; j < n; j += 4) {
    for (int p = 0; p < k; ++p) {
      b_p_j = &B(p, j);
      
      *(packedB_ptr++) = *(b_p_j + 1);
      *(packedB_ptr++) = *(b_p_j + 2);
      *(packedB_ptr++) = *(b_p_j + 3);
      *(packedB_ptr++) = *(b_p_j + 4);
    }
  }

  // A fake packing that demonstrates packing is time-costful
  // double *packedB_t = (double*) malloc(k * n * sizeof(double));
  // double *packedB_ptr_t = packedB_t;
  // double *b_p_j_t;
  // for (int j = 0; j < n; j += 4) {
  //   for (int p = 0; p < k; ++p) {
  //     b_p_j_t = &B(p, j);
      
  //     *(packedB_ptr_t++) = *(b_p_j_t + 1);
  //     *(packedB_ptr_t++) = *(b_p_j_t + 2);
  //     *(packedB_ptr_t++) = *(b_p_j_t + 3);
  //     *(packedB_ptr_t++) = *(b_p_j_t + 4);
  //   }
  // }

  // print_matrix(m, k, A, lda);

  for (int j = 0; j < n; j += 4){
    for (int i = 0; i < m; i += 4){
      c_00_01.reg = _mm_setzero_pd();
      c_10_11.reg = _mm_setzero_pd();
      c_20_21.reg = _mm_setzero_pd();
      c_30_31.reg = _mm_setzero_pd();
      c_02_03.reg = _mm_setzero_pd();
      c_12_13.reg = _mm_setzero_pd();
      c_22_23.reg = _mm_setzero_pd();
      c_32_33.reg = _mm_setzero_pd();


      // b_p0_pntr = &B(0, j);
      // // b_p1_pntr = &B(0, j + 1);
      // b_p2_pntr = &B(0, j + 2); 
      // b_p3_pntr = &B(0, j + 3);

      // avoid to repeat packing A
      // if (j == 0) {
      //   packA(k, &A(i, 0), &packedA[i * k], lda, 4);
      // // packA(k, &A(i, 0), packedA, lda, 4);
      // }
      // printf("packed done, packed sample %f\n", packedA[0]);
      double *packed_b_ptr = &packedB[k * j];    // B(0, j)
      for (int p = 0; p < k; ++p){
        a_0p.reg = _mm_loaddup_pd((double *) &A(i, p));
        a_1p.reg = _mm_loaddup_pd((double *) &A(i + 1, p));
        a_2p.reg = _mm_loaddup_pd((double *) &A(i + 2, p));
        a_3p.reg = _mm_loaddup_pd((double *) &A(i + 3, p));

        // b_p0_p1.reg = _mm_load_pd((double *) b_p0_pntr);
        // b_p2_p3.reg = _mm_load_pd((double *) b_p2_pntr);

        b_p0_p1.reg = _mm_load_pd((double *) packed_b_ptr);
        b_p2_p3.reg = _mm_load_pd((double *) (packed_b_ptr + 2));
        packed_b_ptr += 4;

        /* First colomns and second colomns */
        c_00_01.reg += a_0p.reg * b_p0_p1.reg;
        c_10_11.reg += a_1p.reg * b_p0_p1.reg;

        c_20_21.reg += a_2p.reg * b_p0_p1.reg;
        c_30_31.reg += a_3p.reg * b_p0_p1.reg;

        /* Third colomns fourth colomns */
        c_02_03.reg += a_0p.reg * b_p2_p3.reg;
        c_12_13.reg += a_1p.reg * b_p2_p3.reg;

        c_22_23.reg += a_2p.reg * b_p2_p3.reg;
        c_32_33.reg += a_3p.reg * b_p2_p3.reg;

        // // update b_px_pntr
        // b_p0_pntr += ldb;
        // b_p1_pntr += ldb;
        // b_p2_pntr += ldb; 
        // b_p3_pntr += ldb;
        // printf("p%d\n", p);

      }
      // printf("cece %f", 1000);
      

      C(i, j) += c_00_01.value[0];     C(i, j+1) += c_00_01.value[1];     C(i, j+2) += c_02_03.value[0];     C(i, j+3) += c_02_03.value[1];
      C(i+1, j) += c_10_11.value[0];   C(i+1, j+1) += c_10_11.value[1];   C(i+1, j+2) += c_12_13.value[0];   C(i+1, j+3) += c_12_13.value[1];
      C(i+2, j) += c_20_21.value[0];   C(i+2, j+1) += c_20_21.value[1];   C(i+2, j+2) += c_22_23.value[0];   C(i+2, j+3) += c_22_23.value[1];
      C(i+3, j) += c_30_31.value[0];   C(i+3, j+1) += c_30_31.value[1];   C(i+3, j+2) += c_32_33.value[0];   C(i+3, j+3) += c_32_33.value[1];
    }
  }  
}

// PackA and PackB based on MMult_optim7_1
void MMult_optim7_1_packAB(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  // printf("size %d, %d, %d\n", m, k, n);
  v2d_regv c_00_01, c_10_11, c_20_21, c_30_31,
           c_02_03, c_12_13, c_22_23, c_32_33;
  v2d_regv b_p0_p1, b_p2_p3;
  v2d_regv a_0p, a_1p, a_2p, a_3p;  // the single value will be duplicated to be a vector

  double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  double *packedA = (double*) malloc(m * k * sizeof(double));
  double *packedA_ptr = packedA;
  for (int i = 0; i < m; i += 4) {
    for (int p = 0; p < k; ++p) {
      *packedA_ptr = A(i, p);   // ATTENTION!: not A[i, p]
      *(packedA_ptr + 1) = A(i + 1, p);
      *(packedA_ptr + 2) = A(i + 2, p);
      *(packedA_ptr + 3) = A(i + 3, p);

      packedA_ptr += 4;
    }
  }

  double *packedB = (double*) malloc(k * n * sizeof(double));
  double *packedB_ptr = packedB;
  for (int j = 0; j < n; j += 4) {
    for (int p = 0; p < k; ++p) {
      *packedB_ptr = B(p, j);
      *(packedB_ptr + 1) = B(p, j + 1);
      *(packedB_ptr + 2) = B(p, j + 2);
      *(packedB_ptr + 3) = B(p, j + 3);

      packedB_ptr += 4;
    }
  }

  // print_matrix(m, k, A, lda);

  for (int j = 0; j < n; j += 4){
    for (int i = 0; i < m; i += 4){
      c_00_01.reg = _mm_setzero_pd();
      c_10_11.reg = _mm_setzero_pd();
      c_20_21.reg = _mm_setzero_pd();
      c_30_31.reg = _mm_setzero_pd();
      c_02_03.reg = _mm_setzero_pd();
      c_12_13.reg = _mm_setzero_pd();
      c_22_23.reg = _mm_setzero_pd();
      c_32_33.reg = _mm_setzero_pd();


      // b_p0_pntr = &B(0, j);
      // // b_p1_pntr = &B(0, j + 1);
      // b_p2_pntr = &B(0, j + 2); 
      // b_p3_pntr = &B(0, j + 3);

      // avoid to repeat packing A
      // if (j == 0) {
      //   packA(k, &A(i, 0), &packedA[i * k], lda, 4);
      // // packA(k, &A(i, 0), packedA, lda, 4);
      // }
      // printf("packed done, packed sample %f\n", packedA[0]);
      double *packed_a_ptr = &packedA[i * k];  // A(i 0)
      double *packed_b_ptr = &packedB[k * j];    // B(0, j)

      for (int p = 0; p < k; ++p){
        // a_0p.reg = _mm_loaddup_pd((double *) &A(i, p));
        // a_1p.reg = _mm_loaddup_pd((double *) &A(i + 1, p));
        // a_2p.reg = _mm_loaddup_pd((double *) &A(i + 2, p));
        // a_3p.reg = _mm_loaddup_pd((double *) &A(i + 3, p));

        a_0p.reg = _mm_loaddup_pd((double *) packed_a_ptr);  
        a_1p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 1));
        a_2p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 2));
        a_3p.reg = _mm_loaddup_pd((double *) (packed_a_ptr + 3));
        packed_a_ptr += 4;

        b_p0_p1.reg = _mm_load_pd((double *) packed_b_ptr);
        b_p2_p3.reg = _mm_load_pd((double *) (packed_b_ptr + 2));
        packed_b_ptr += 4;

        /* First colomns and second colomns */
        c_00_01.reg += a_0p.reg * b_p0_p1.reg;
        c_10_11.reg += a_1p.reg * b_p0_p1.reg;

        c_20_21.reg += a_2p.reg * b_p0_p1.reg;
        c_30_31.reg += a_3p.reg * b_p0_p1.reg;

        /* Third colomns fourth colomns */
        c_02_03.reg += a_0p.reg * b_p2_p3.reg;
        c_12_13.reg += a_1p.reg * b_p2_p3.reg;

        c_22_23.reg += a_2p.reg * b_p2_p3.reg;
        c_32_33.reg += a_3p.reg * b_p2_p3.reg;

        // // update b_px_pntr
        // b_p0_pntr += ldb;
        // b_p1_pntr += ldb;
        // b_p2_pntr += ldb; 
        // b_p3_pntr += ldb;
        // printf("p%d\n", p);

      }
      // printf("cece %f", 1000);
      

      C(i, j) += c_00_01.value[0];     C(i, j+1) += c_00_01.value[1];     C(i, j+2) += c_02_03.value[0];     C(i, j+3) += c_02_03.value[1];
      C(i+1, j) += c_10_11.value[0];   C(i+1, j+1) += c_10_11.value[1];   C(i+1, j+2) += c_12_13.value[0];   C(i+1, j+3) += c_12_13.value[1];
      C(i+2, j) += c_20_21.value[0];   C(i+2, j+1) += c_20_21.value[1];   C(i+2, j+2) += c_22_23.value[0];   C(i+2, j+3) += c_22_23.value[1];
      C(i+3, j) += c_30_31.value[0];   C(i+3, j+1) += c_30_31.value[1];   C(i+3, j+2) += c_32_33.value[0];   C(i+3, j+3) += c_32_33.value[1];
    }
  }  
}


// use more flexible block size for m, n and k, seperately
void MMult_optim9_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{    
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
    int mSize = MIN(mBlockSize, m - mBlockStart);
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
      int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize

      for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
        int nSize = MIN(nBlockSize, n - nBlockStart);
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1_packA(mSize, kSize, nSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
      }
    }
  }
}

void MMult_optim9_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{    
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
    int mSize = MIN(mBlockSize, m - mBlockStart);
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
      int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize

      for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
        int nSize = MIN(nBlockSize, n - nBlockStart);
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1_packB(mSize, kSize, nSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
      }
    }
  }
}

void MMult_optim9_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{    
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
    int mSize = MIN(mBlockSize, m - mBlockStart);
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
      int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize

      for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
        int nSize = MIN(nBlockSize, n - nBlockStart);
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1_packAB(mSize, kSize, nSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
      }
    }
  }
}




// // TODO: why pack need Z or N style? https://zhuanlan.zhihu.com/p/69700540
// void packA(int mBlock, int kBlock, double *A, double *packedA, int lda) {  // A here is one block of full A 
//     for (int i = 0; i < mBlock; ++i) {
//         double *a_i0_ptr = &A(i, 0);
//         for (int j = 0; j < kBlock; ++j) {
//             *packedA = *(a_i0_ptr + j);
//             packedA += 1;
//         }
//     }
// }

// void packB(int kBlock, int nBlock, double *B, double *packedB, int ldb) {  // B here is one block of full B 
//     for (int i = 0; i < kBlock; ++i) {
//         double *b_i0_ptr = &B(i, 0);
//         for (int j = 0; j < nBlock; ++j) {
//             *packedB = *(b_i0_ptr + j);
//             packedB += 1;
//         }
//     }
// }

// // pack A for MMult_optim1_1
// // a little boost
// void MMult_optim9_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_base(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
//       }
//     }
//   }
// }

// // pack B for MMult_optim1_1
// // slightly faster than MMult_optim9_1
// // I think it is because indexing B encounters more cashe miss than indexding A, so packing for B helps more
// void MMult_optim9_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
//       }
//     }
//   }
// }

// // pack A and B for MMult_optim1_1
// // slower than MMult_optim9_2, faster than MMult_optim9_1 (confusing)
// void MMult_optim9_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockA[blockSize * blockSize];   // mBlock * kBlock
//   double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
//           packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_base(blockSize, blockSize, blockSize, packedBlockA, packedBlockB, &C(mBlockStart, nBlockStart), blockSize, blockSize, ldc);
//       }
//     }
//   }
// }

// // pack A for MMult_optim8_7
// // slower than MMult_optim8_7
// void MMult_optim9_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
//       }
//     }
//   }
// }

// // pack B for MMult_optim8_7
// // sligtly slower than MMult_optim8_7
// void MMult_optim9_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_optim7_1(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
//       }
//     }
//   }
// }

// // pack A and B for MMult_optim8_7
// // similar to MMult_optim9_4 (slower than MMult_optim8_7)
// void MMult_optim9_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockA[blockSize * blockSize];   // mBlock * kBlock
//   double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//           packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
//           packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//           MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, packedBlockB, &C(mBlockStart, nBlockStart), blockSize, blockSize, ldc);
//       }
//     }
//   }
// }

// // Remove the repeated packing of A in MMult_optim9_4
// // similar to MMult_optim8_7 (packing for A is less meaningful?)
// void MMult_optim9_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//       packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
//       for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//         //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//         MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
//       }
//     } 
//   }
// }

// // Remove the repeated packing of B in MMult_optim9_5
// // nearly the same with MMult_optim8_7 (packing for B is also less meaningful?)
// void MMult_optim9_8(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

//   for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//     for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//       packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
//       for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//       //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//         MMult_optim7_1(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
//       }
//     }
//   }
// }

// // use more flexible block size for m, n and k, seperately
// // PackB
// int mBlockSize = 256, kBlockSize = 128, nBlockSize = 40;
// void MMult_optim9_9(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {  
//   for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
//     int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
//       int nSize = MIN(nBlockSize, n - nBlockStart);

//       double packedBlockB[kSize * nSize]; 
//       packB(kSize, nSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);

//       for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
//         int mSize = MIN(mBlockSize, m - mBlockStart);
//       //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//         MMult_optim7_1(mSize, kSize, nSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, nSize, ldc);
//       }
//     }
//   }
// }

// // use more flexible block size for m, n and k, seperately
// // PackA
// // int mBlockSize = 256, kBlockSize = 128, nBlockSize = 40;
// void MMult_optim9_10(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {    
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
//     int mSize = MIN(mBlockSize, m - mBlockStart);
//     for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
//       int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize

//       double packedBlockA[mSize * kSize];
//       packA(mSize, kSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);

//       for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
//         int nSize = MIN(nBlockSize, n - nBlockStart);
//       //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//         MMult_optim7_1(mSize, kSize, nSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), kSize, ldb, ldc);
//       }
//     }
//   }
// }




