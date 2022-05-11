// #include "params.h"
// #include "MMult.h"

// // Use cache blocking 
// // Not exactly the same with:
// // https://github.com/flame/how-to-optimize-gemm/blob/master/src/MMult_4x4_11.c

// // How it works
// // 1. https://www.youtube.com/watch?v=G92BCtfTwOE (at time 5:00)
// // 2. https://stackoverflow.com/questions/63614160/how-does-cache-blocking-actually-speed-up-performance
// // 3. https://zhuanlan.zhihu.com/p/69700540 [Recommanded]


// int blockSize;
// blockSize = 40;

// // ======= 8_0 ~ 8_3, ONLY block m and n (no performance gain) =======> // 
// // block m, n for pure MMult_base
// // No performance gain 
// void MMult_optim8_0(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {

//       for (int i = mBlockStart; i < mBlockStart + blockSize; ++i){
//         for (int j = nBlockStart; j < nBlockStart + blockSize; ++j){
//           for (int p = 0; p < k; ++p)
//             C(i, j) += A(i, p) * B(p, j);
//         }
//       }

//     }
//   }
// }

// // block m, n for pure MMult_base (functional style)
// // No performance gain
// void MMult_optim8_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       MMult_base(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//     }
//   }
// }

// // block m, n for MMult_optim1_1
// // even slower (confusing)
// void MMult_optim8_2(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       MMult_optim1_1(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//     }
//   }
// }

// // block m, n for MMult_optim7_1
// // No performance gain 
// void MMult_optim8_3(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       MMult_optim7_1(blockSize, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//     }
//   }
// }
// // <======= 8_0 ~ 8_3, ONLY block m and n (no performance gain) ======= // 


// // ======= 8_4 ~ 8_7, ONLY block m, n and k (performance boost for large matrix is maintained) =======> // 
// // block m, n and k for pure MMult_base
// // performance boost for large matrix is maintained
// void MMult_optim8_4(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < n; kBlockStart += blockSize) {

//         for (int i = mBlockStart; i < mBlockStart + blockSize; ++i){
//           for (int j = nBlockStart; j < nBlockStart + blockSize; ++j){
//             for (int p = kBlockStart; p < kBlockStart + blockSize; ++p)
//               C(i, j) += A(i, p) * B(p, j);
//           }
//         }
//       }

//     }
//   }
// }

// // block m, n and k for pure MMult_base (functional style)
// // performance boost for large matrix is maintained
// void MMult_optim8_5(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//         MMult_base_k_seg(blockSize, kBlockStart, kBlockStart + blockSize, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//       }
//     }
//   }
// }

// // avoid using *_k_seg() function for compatibility
// void MMult_optim8_5_1(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//         MMult_base(blockSize, blockSize, blockSize, &A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//       }
//     }
//   }
// }


// // block m, n and k for pure MMult_optim1_1
// // performance boost for large matrix is maintained
// void MMult_optim8_6(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//         MMult_optim1_1_k_seg(blockSize, kBlockStart, kBlockStart + blockSize, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//       }
//     }
//   }
// }

// // block m, n and k for pure MMult_optim7_1
// // performance boost for large matrix is maintained
// // Well done!
// void MMult_optim8_7(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
// {
//   for (int mBlockStart = 0; mBlockStart < m; mBlockStart += blockSize) {
//     for (int nBlockStart = 0; nBlockStart < n; nBlockStart += blockSize) {
//       for (int kBlockStart = 0; kBlockStart < k; kBlockStart += blockSize) {
//         MMult_optim7_1_k_seg(blockSize, kBlockStart, kBlockStart + blockSize, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//       }
//       // MMult_optim7_1_k_seg(blockSize, 0, k, blockSize, &A(mBlockStart, 0), &B(0, nBlockStart), &C(mBlockStart, nBlockStart), lda, ldb, ldc);
//     }
//   }
// }