#include "params.h"
#include "utils.h"
#include "MMult.h"

// Note that CUDA index elements by (x, y), rather than (i, j)

// one element in C per thread
__global__ void gemm_base(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < m) {
        for (int i = 0; i < k; ++i) {
            d_C(x, y) += d_A(x, i) * d_B(i, y);
        }
    }
}


void MMult_base(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((m + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    gemm_base<<<numBlocks, threadsPerBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}
