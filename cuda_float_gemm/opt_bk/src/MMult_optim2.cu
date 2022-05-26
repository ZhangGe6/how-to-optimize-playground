#include "params.h"
#include "utils.h"
#include "MMult.h"

// m, n in GPU memory is naturally blocked, so what we need to do is move the blocked data from global memory to shared memory
// BLOCK_SIZE * BLOCK_SIZE threads are allocated to calculate the sub BLOCK_SIZE * BLOCK_SIZE matrix in C
// we also split k (with blockDim.x here) to fit the blocked sub A and sub B into shared memory
// If the code is hard to read, please refer to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// for more readable version (the official code is yyds)
template <int BLOCK_SIZE> 
__global__ void gemm_optim2_1(int m, int k, int n,
     float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc
    ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    // Each thread computes one element of Csub
    // by accumulating results into C_value
    float C_value = 0;

    // 各线程依次完成“千层饼”的第tile_k_id * BLOCK_SIZE~ tile_k_id * BLOCK_SIZE + BLOCK_SIZE层，并叠加
    // printf("k %d, BLOCK_SIZE %d\n", k, BLOCK_SIZE);
    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {
        // printf("tile_k_id %d\n", tile_k_id);
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
            C_value += A_shared[row_in_block][i] * B_shared[i][col_in_block];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    d_C(row, col) = C_value;
}

void MMult_optim2_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    const int BLOCK_SIZE = 16;
    // const int BLOCK_SIZE = 64; // TODO: why 64 here not work? 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // printf("(m + BLOCK_SIZE - 1) / BLOCK_SIZE %d\n", (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // printf("k / BLOCK_SIZE %d\n", k / BLOCK_SIZE);

    gemm_optim2_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}

