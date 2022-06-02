#include "params.h"
#include "utils.h"
#include "MMult.h"

// thread accumulates the result of each of these products into a register and once done writes the result to global memory
// as https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory illustrates
// doubles the speed!!!
__global__ void gemm_kernel_optim1_1(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float C_value = 0;
    for (int i = 0; i < K; ++i) {
        // d_C(row, col) += d_A(row, i) * d_B(i, col);
        C_value += A[OFFSET(row, i, K)] * B[OFFSET(i, col, N)];
    }
    C[OFFSET(row, col, N)] = C_value;
}


void MMult_optim1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);  // threadsPerBlock
    dim3 dimGrid((M + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);  // numBlocks

    gemm_kernel_optim1_1<<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
}