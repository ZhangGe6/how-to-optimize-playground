#include "params.h"
#include "utils.h"
#include "MMult.h"

// block the whole matrix into shared memory
// square block now

// If the code is hard to read, please refer to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// for more readable version (the official code is yyds)
template <int BLOCK_SIZE> 
__global__ void gemm_kernel_optim2_1(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    float C_value = 0;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {
        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = A + blockIdx.y * (BLOCK_SIZE * K) + tile_idx;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = B + tile_idx * N + blockIdx.x * BLOCK_SIZE;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        A_shared[thread_row][thread_col] = Asub[OFFSET(thread_row, thread_col, K)];
        B_shared[thread_row][thread_col] = Bsub[OFFSET(thread_row, thread_col, N)];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value += A_shared[thread_row][i] * B_shared[i][thread_col];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    // d_C(row, col) = C_value;
    C[OFFSET(row, col, N)] = C_value;
}

void MMult_optim2_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    const int BLOCK_SIZE = 16;
    // const int BLOCK_SIZE = 64; // TODO: why 64 here not work? 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // printf("(m + BLOCK_SIZE - 1) / BLOCK_SIZE %d\n", (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // printf("k / BLOCK_SIZE %d\n", k / BLOCK_SIZE);

    gemm_kernel_optim2_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
}

