#include "params.h"
#include "utils.h"
#include "MMult.h"

// Note that CUDA index elements by (x, y), rather than (i, j)

// thread accumulates the result of each of these products into a register and once done writes the result to global memory
// as https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory illustrates
// doubles the speed!!!
__global__ void gemm_optim1_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= m) return;
    
    float C_xy = 0;
    for (int i = 0; i < k; ++i) {
        C_xy += d_A(x, i) * d_B(i, y);
    }
    d_C(x, y) = C_xy;
}

void MMult_optim1_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((m + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    gemm_optim1_1<<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}



// m, n in GPU memory is naturally blocked, so what we need to do is move the blocked data from global memory to shared memory
// BLOCK_SIZE * BLOCK_SIZE threads are allocated to calculate the sub BLOCK_SIZE * BLOCK_SIZE matrix in C
// we also split k (with blockDim.x here) to fit the blocked sub A and sub B into shared memory
// If the code is hard to read, please refer to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// for more readable version (the official code is yyds)
template <int BLOCK_SIZE> 
__global__ void gemm_optim1_2(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= m) return;

    // Each thread computes one element of Csub
    // by accumulating results into C_xy
    float C_xy = 0;

    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {  // 各线程依次完成“千层饼”的第tile_k_id层，并叠加
        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE * k) + tile_k_id * BLOCK_SIZE;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE * n) + blockIdx.x * BLOCK_SIZE;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        A_shared[row_in_block][col_in_block] = Asub[row_in_block * lda + col_in_block];
        B_shared[row_in_block][col_in_block] = Bsub[row_in_block * ldb + col_in_block];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_xy += A_shared[row_in_block][i] * B_shared[i][col_in_block];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    // printf("C[%d][%d]: %f\n", y, x, C_xy);
    d_C(x, y) = C_xy;
}

void MMult_optim1_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_optim1_2<BLOCK_SIZE><<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}

