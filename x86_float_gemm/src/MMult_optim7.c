#include "params.h"
#include "MMult.h"

void MMult_optim7_1(float *A, float *B, float *C, int M, int K, int N, int lda, int ldb, int ldc)
{
    int BLOCKSIZE = 40;
    for (int mBlockStart = 0; mBlockStart < M; mBlockStart += BLOCKSIZE) {
        for (int nBlockStart = 0; nBlockStart < N; nBlockStart += BLOCKSIZE) {
            for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BLOCKSIZE) {
                // printf("mBlockStart %d, nBlockStart %d, kBlockStart %d\n", mBlockStart, nBlockStart, kBlockStart);
                // MMult_base(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);

                // MMult_optim3_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
                MMult_optim4_1(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);

                // MMult_optim3_4(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
                // MMult_optim4_2(&A(mBlockStart, kBlockStart), &B(kBlockStart, nBlockStart), &C(mBlockStart, nBlockStart), BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, lda, ldb, ldc);
            }
        }
    }
}