#include "params.h"
#include "utils.h"
#include "MMult.h"

// one element in C per thread
__global__ void gemm_kernel_base(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    for (int i = 0; i < K; ++i) {
        // d_C(row, col) += d_A(row, i) * d_B(i, col);
        C[OFFSET(row, col, N)] += A[OFFSET(row, i, K)] * B[OFFSET(i, col, N)];
    }
}

void MMult_base(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);  // threadsPerBlock
    dim3 dimGrid((M + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);  // numBlocks

    gemm_kernel_base<<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
}

// 这种做法，线程在运算时，每个数据的读和写都直接和global memory进行，有很长的时延

// int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
