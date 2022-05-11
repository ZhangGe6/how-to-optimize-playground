#include "params.h"
#include "MMult.h"

// Packing into contiguous memory
// Not exactly the same with:
// https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_12.c

// How it works
// 1. https://chhzh123.github.io/blogs/2020-03-20-tvm-gemm/#%E6%95%B0%E7%BB%84%E6%89%93%E5%8C%85packing

int blockSize = 40;

// TODO: why pack need Z or N style? https://zhuanlan.zhihu.com/p/69700540
void packA(int mBlock, int kBlock, double *A, double *packedA, int lda) {  // A here is one block of full A 
    for (int i = 0; i < mBlock; ++i) {
        double *a_i0_ptr = &A(i, 0);
        for (int j = 0; j < kBlock; ++j) {
            *packedA = *(a_i0_ptr + j);
            packedA += 1;
        }
    }
}

void packB(int kBlock, int nBlock, double *B, double *packedB, int ldb) {  // B here is one block of full B 
    for (int i = 0; i < kBlock; ++i) {
        double *b_i0_ptr = &B(i, 0);
        for (int j = 0; j < nBlock; ++j) {
            *packedB = *(b_i0_ptr + j);
            packedB += 1;
        }
    }
}


// pack A for MMult_optim1_1
// a little boost
void MMult_optim9_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_base(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
      }
    }
  }
}

// pack B for MMult_optim1_1
// slightly faster than MMult_optim9_1
// I think it is because indexing B encounters more cashe miss than indexding A, so packing for B helps more
void MMult_optim9_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
      }
    }
  }
}

// pack A and B for MMult_optim1_1
// slower than MMult_optim9_2, faster than MMult_optim9_1 (confusing)
void MMult_optim9_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockA[blockSize * blockSize];   // mBlock * kBlock
  double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
          packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_base(blockSize, blockSize, blockSize, packedBlockA, packedBlockB, &C(mBlockStart, nBlockStart), blockSize, blockSize, ldc);
      }
    }
  }
}

// pack A for MMult_optim8_7
// slower than MMult_optim8_7
void MMult_optim9_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
      }
    }
  }
}

// pack B for MMult_optim8_7
// sligtly slower than MMult_optim8_7
void MMult_optim9_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_optim7_1(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
      }
    }
  }
}

// pack A and B for MMult_optim8_7
// similar to MMult_optim9_4 (slower than MMult_optim8_7)
void MMult_optim9_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockA[blockSize * blockSize];   // mBlock * kBlock
  double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
      for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
          packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
          packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
          MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, packedBlockB, &C(mBlockStart, nBlockStart), blockSize, blockSize, ldc);
      }
    }
  }
}

// Remove the repeated packing of A in MMult_optim9_4
// similar to MMult_optim8_7 (packing for A is less meaningful?)
void MMult_optim9_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockA[blockSize * blockSize];   // mBlock * kBlock

  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
      packA(blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);
      for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
        //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1(blockSize, blockSize, blockSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, ldb, ldc);
      }
    } 
  }
}

// Remove the repeated packing of B in MMult_optim9_5
// nearly the same with MMult_optim8_7 (packing for B is also less meaningful?)
void MMult_optim9_8(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  double packedBlockB[blockSize * blockSize];   // kBlock * nBlock

  for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
      packB(blockSize, blockSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);
      for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, blockSize, ldc);
      }
    }
  }
}

// use more flexible block size for m, n and k, seperately
// PackB
int mBlockSize = 256, kBlockSize = 128, nBlockSize = 40;
void MMult_optim9_9(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{  
  for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
    int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize
    for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
      int nSize = MIN(nBlockSize, n - nBlockStart);

      double packedBlockB[kSize * nSize]; 
      packB(kSize, nSize, &B(kBlockStart, nBlockStart), packedBlockB, ldb);

      for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
        int mSize = MIN(mBlockSize, m - mBlockStart);
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1(mSize, kSize, nSize, &A(mBlockStart, kBlockStart), packedBlockB, &C(mBlockStart, nBlockStart), lda, nSize, ldc);
      }
    }
  }
}

// use more flexible block size for m, n and k, seperately
// PackA
// int mBlockSize = 256, kBlockSize = 128, nBlockSize = 40;
void MMult_optim9_10(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{    
  for (int mBlockStart = 0; mBlockStart < m; mBlockStart += mBlockSize) {
    int mSize = MIN(mBlockSize, m - mBlockStart);
    for (int kBlockStart = 0; kBlockStart < k; kBlockStart += kBlockSize) {
      int kSize = MIN(kBlockSize, k - kBlockStart);   // in case the left k dimension size smaller than kBlockSize

      double packedBlockA[mSize * kSize];
      packA(mSize, kSize, &A(mBlockStart, kBlockStart), packedBlockA, lda);

      for (int nBlockStart = 0; nBlockStart < n; nBlockStart += nBlockSize) {
        int nSize = MIN(nBlockSize, n - nBlockStart);
      //   MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
        MMult_optim7_1(mSize, kSize, nSize, packedBlockA, &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), kSize, ldb, ldc);
      }
    }
  }
}

