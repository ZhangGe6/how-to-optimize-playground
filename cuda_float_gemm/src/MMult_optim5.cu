#include "params.h"
#include "utils.h"
#include "MMult.h"

// use more flexible block size for m, n and k based on MMult_optim3_2
template<
    int BLOCK_SIZE_M, 
    int BLOCK_SIZE_N, 
    int BLOCK_SIZE_K, 
    int ELE_PER_THREAD_ROW, 
    int ELE_PER_THREAD_COL
> 
__global__ void gemm_optim5_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;

    // by accumulating results into C_value
    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {0};

    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE_K); ++tile_k_id) {
        __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
        __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE_M * k) + tile_k_id * BLOCK_SIZE_K;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE_K * n) + blockIdx.x * BLOCK_SIZE_N;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // :star: use thread.y, thread.x to map the location of shared memory
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            #pragma unroll
            for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
                int row_in_shared = row_in_block * ELE_PER_THREAD_ROW + row_offset;
                int col_in_shared = col_in_block * ELE_PER_THREAD_COL + col_offset;

                A_shared[row_in_shared][col_in_shared] = Asub[row_in_shared * lda + col_in_shared];
                B_shared[row_in_shared][col_in_shared] = Bsub[row_in_shared * ldb + col_in_shared];
            }
        }
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // each thread is responsible for an `expanded` area
        // [row_in_block * ELE_PER_THREAD_ROW:(row_in_block + 1) * ELE_PER_THREAD_ROW][col_in_block * ELE_PER_THREAD_COL:(col_in_block + 1) * ELE_PER_THREAD_COL]
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            #pragma unroll
            for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
                int row_in_shared = row_in_block * ELE_PER_THREAD_ROW + row_offset;
                int col_in_shared = col_in_block * ELE_PER_THREAD_COL + col_offset;
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE_K; ++i) {
                    // register level writing
                    C_value[row_offset][col_offset] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared];
                }
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    // let's comprehend the following code in a global C matrix view 
    // :star: use row and col to map the location of global memory
    #pragma unroll
    for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
        #pragma unroll
        for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
            int row_in_C = row * ELE_PER_THREAD_ROW + row_offset;
            int col_in_C = col * ELE_PER_THREAD_COL + col_offset;
            
            d_C[row_in_C * ldc + col_in_C] = C_value[row_offset][col_offset];   // d_C(row_in_C, col_in_C)
        }
    }

}

void MMult_optim5_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 32;
    const int ELE_PER_THREAD_ROW = 4;
    const int ELE_PER_THREAD_COL = 4;
    dim3 dimBlock(BLOCK_SIZE_N / ELE_PER_THREAD_COL, BLOCK_SIZE_M / ELE_PER_THREAD_ROW);
    dim3 dimGrid((m + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (n + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_optim5_1<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
