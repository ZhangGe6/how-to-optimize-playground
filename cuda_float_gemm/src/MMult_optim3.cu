#include "params.h"
#include "utils.h"
#include "MMult.h"

// every thread compute more elements
// How can it speedup: https://cnugteren.github.io/tutorial/pages/page5.html

// every thread compute 4 (consecutive) elements in x dimension, this is a demo
// https://github1s.com/Cjkkkk/CUDA_gemm/blob/HEAD/src/cuda/dense_legacy.cu#L215-L216
// fairly speedup
template <int BLOCK_SIZE> 
__global__ void gemm_optim3_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;

    // Each thread computes 4 element of Csub
    // by accumulating results into C_value
    float C_value[4] = {0, 0, 0, 0};
    
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
            
    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {

        // Get sub-(block)-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE * k) + tile_k_id * BLOCK_SIZE;  
        // Get sub-(block)-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE * n) + blockIdx.x * BLOCK_SIZE;  

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads 4 element of each sub-matrix (across 4 consecutive colomns)
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        A_shared[row_in_block][4 * col_in_block] = Asub[row_in_block * lda + 4 * col_in_block];
        A_shared[row_in_block][4 * col_in_block + 1] = Asub[row_in_block * lda + 4 * col_in_block + 1];
        A_shared[row_in_block][4 * col_in_block + 2] = Asub[row_in_block * lda + 4 * col_in_block + 2];
        A_shared[row_in_block][4 * col_in_block + 3] = Asub[row_in_block * lda + 4 * col_in_block + 3];

        B_shared[row_in_block][4 * col_in_block] = Bsub[row_in_block * ldb + 4 * col_in_block];
        B_shared[row_in_block][4 * col_in_block + 1] = Bsub[row_in_block * ldb + 4 * col_in_block + 1];
        B_shared[row_in_block][4 * col_in_block + 2] = Bsub[row_in_block * ldb + 4 * col_in_block + 2];
        B_shared[row_in_block][4 * col_in_block + 3] = Bsub[row_in_block * ldb + 4 * col_in_block + 3];
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        // Each thread calculate 4 element of each sub-matrix (1x4 consecutive colomns)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value[0] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block];
            C_value[1] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 1];
            C_value[2] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 2];
            C_value[3] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 3];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    // d_C(row, col) = C_value;
    d_C(row, 4 * col) = C_value[0];
    d_C(row, 4 * col + 1) = C_value[1];
    d_C(row, 4 * col + 2) = C_value[2];
    d_C(row, 4 * col + 3) = C_value[3];
}

void MMult_optim3_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE);   // decrease the thread dim along x dim by 4
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_optim3_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}

// every thread compute multiple (consecutive) elements in both x and y dimensions 
// a large speedup
// and `#pragma unroll` really matters here
// About how `#pragma unroll`work, see
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll

template <int BLOCK_SIZE, int ELE_PER_THREAD_ROW, int ELE_PER_THREAD_COL> 
__global__ void gemm_optim3_2(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;

    // by accumulating results into C_value
    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {0};

    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {
        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE * k) + tile_k_id * BLOCK_SIZE;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE * n) + blockIdx.x * BLOCK_SIZE;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads 4 element of each sub-matrix (across 4 strided colomns)
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
                for (int i = 0; i < BLOCK_SIZE; ++i) {
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

void MMult_optim3_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // params really matters:
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 4, 4 => ~2000GFLOPs
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 8, 8 => ~200GFLOPs
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 32, 4, 4 => ~1300GFLOPs

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    const int ELE_PER_THREAD_ROW = 4;
    const int ELE_PER_THREAD_COL = 4;
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / ELE_PER_THREAD_COL, BLOCK_SIZE / ELE_PER_THREAD_ROW);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_optim3_2<BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


