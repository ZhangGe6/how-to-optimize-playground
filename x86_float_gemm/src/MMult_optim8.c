#include "params.h"
#include "MMult.h"

// use cache blocking

void MMult_optim8_1(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
    int blockSize = 40;
    for (int mBlockStart = 0; mBlockStart < M; mBlockStart += blockSize) {
        for (int nBlockStart = 0; nBlockStart < N; nBlockStart += blockSize) {
            for (int kBlockStart = 0; kBlockStart < K; kBlockStart += blockSize) {
                // printf("mBlockStart %d, nBlockStart %d, kBlockStart %d\n", mBlockStart, nBlockStart, kBlockStart);
                // MMult_base(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);

                // ===== MMult_optim4_1 will suffer from floating-point rounding error, and MMult_optim4_1_1 will not. ====== // 
                // MMult_optim3_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
                // MMult_optim4_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
                // MMult_optim4_1_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);

                // ===== MMult_optim4_2 will suffer from floating-point rounding error, and MMult_optim4_2_1 will not. ====== // 
                // MMult_optim3_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
                // MMult_optim4_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
                // MMult_optim4_2_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);

                MMult_optim6_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), blockSize, blockSize, blockSize, lda, ldb, ldc);
            }
        }
    }
}

// use more flexible blockSize
void MMult_optim8_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
{
    int mBlockSize = 80, nBlockSize = 80, kBlockSize = 100;
    for (int mBlockStart = 0; mBlockStart < M; mBlockStart += mBlockSize) {
        int mSize = MIN(mBlockSize, M - mBlockStart);     // in case the left m dimension size smaller than mBlockSize
        for (int nBlockStart = 0; nBlockStart < N; nBlockStart += nBlockSize) {
            int nSize = MIN(nBlockSize, N - nBlockStart);
            for (int kBlockStart = 0; kBlockStart < K; kBlockStart += kBlockSize) {
                int kSize = MIN(kBlockSize, K - kBlockStart);
                // MMult_optim6_2(float *A, float *B, float *C, const int M, const int K, const int N, const int lda, const int ldb, const int ldc)
                MMult_optim6_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), mSize, kSize, nSize, lda, ldb, ldc);
            }
        }
    }
}
