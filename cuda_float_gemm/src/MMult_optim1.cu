#include "params.h"
#include "utils.h"
#include "MMult.h"

// Note that CUDA index elements by (x, y), rather than (i, j)

// thread accumulates the result of each of these products into a register and once done writes the result to global memory
// as https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory illustrates
// doubles the speed!!!
__global__ void gemm_optim1_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float C_value = 0;
    for (int i = 0; i < k; ++i) {
        C_value += d_A(row, i) * d_B(i, col);
    }
    d_C(row, col) = C_value;
}

void MMult_optim1_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((m + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    gemm_optim1_1<<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}



