#include "params.h"
#include "utils.h"
#include "MMult.h"

// optimzation gym for square block optimzation

// use direct indexing based on gemm_kernel_optim6_1
// performance drop from 3800 to 3400..., why? (see also MMult_optim_rect1_1_1)
template <int BLOCK_SIZE, int ELE_PER_THREAD_ROW, int ELE_PER_THREAD_COL> 
__global__ void gemm_kernel_optim6_1_1(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // float4 C_value[ELE_PER_THREAD_ROW] = {{0, 0, 0, 0}};
    // ELE_PER_THREAD_COL == 4 here
    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {{0, 0, 0, 0}};
    // registers for A and B
    float frag_a[ELE_PER_THREAD_ROW];
    float frag_b[ELE_PER_THREAD_COL];

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {
        // =================== exp here ===================> // 
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads (ELE_PER_THREAD_ROW * ELE_PER_THREAD_COL) `consective` element of each sub-matrix
        // :star: use thread.y, thread.x to map the location of shared memory
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
            FETCH_FLOAT4(A_shared[row_in_shared][4 * thread_col]) = FETCH_FLOAT4(A[OFFSET(
                                                                        blockIdx.y * BLOCK_SIZE + row_in_shared, // row
                                                                        4 * thread_col,                                // col
                                                                        K)]);
            FETCH_FLOAT4(B_shared[row_in_shared][4 * thread_col]) = FETCH_FLOAT4(B[OFFSET(tile_idx + row_in_shared, // row
                                                                        blockIdx.x * BLOCK_SIZE + 4 * thread_col, // col
                                                                        N)]);
        }
        // <=================== exp here =================== // 
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // each thread is responsible for an `expanded` area
        // [thread_row * ELE_PER_THREAD_ROW:(thread_row + 1) * ELE_PER_THREAD_ROW][thread_col * ELE_PER_THREAD_COL:(thread_col + 1) * ELE_PER_THREAD_COL]
        // ========== v1 ========== //
        // for (int i = 0; i < BLOCK_SIZE; ++i) {
        //     // load A from shared memory to register
        //     #pragma unroll
        //     for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
        //         frag_a[row_offset] = A_shared[thread_row * ELE_PER_THREAD_ROW + row_offset][i];
        //     }

        //     // load B from shared memory to register
        //     #pragma unroll
        //     for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
        //         frag_b[col_offset] = B_shared[i][thread_col * ELE_PER_THREAD_COL + col_offset];
        //     }
            
        //     #pragma unroll
        //     for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
        //         #pragma unroll
        //         for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
        //             C_value[row_offset][col_offset] += frag_a[row_offset] * frag_b[col_offset];
        //         }
        //     }
        // }

        // ========== v2 ========== //
        // faster than v1, similer to MMult_optim5_2
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            // load A from shared memory to register
            #pragma unroll
            for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
                frag_a[row_offset] = A_shared[thread_row * ELE_PER_THREAD_ROW + row_offset][i];
            }

            // load B from shared memory to register
            frag_b[0] = B_shared[i][thread_col * ELE_PER_THREAD_COL];
            frag_b[1] = B_shared[i][thread_col * ELE_PER_THREAD_COL + 1];
            frag_b[2] = B_shared[i][thread_col * ELE_PER_THREAD_COL + 2];
            frag_b[3] = B_shared[i][thread_col * ELE_PER_THREAD_COL + 3];
            
            #pragma unroll
            for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
                C_value[row_offset][0] += frag_a[row_offset] * frag_b[0];
                C_value[row_offset][1] += frag_a[row_offset] * frag_b[1];
                C_value[row_offset][2] += frag_a[row_offset] * frag_b[2];
                C_value[row_offset][3] += frag_a[row_offset] * frag_b[3];
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
        int row_in_C = row * ELE_PER_THREAD_ROW + row_offset;
        int col_in_C = col * 4;
        FETCH_FLOAT4(C[OFFSET(row_in_C, col_in_C, N)]) = FETCH_FLOAT4(C_value[row_offset]);   // d_C(row_in_C, col_in_C)   
    }

}

void MMult_optim6_1_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    // params really matters:
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 4, 4 => ~2000GFLOPs
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 8, 8 => ~200GFLOPs
    // BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 32, 4, 4 => ~1300GFLOPs

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    const int ELE_PER_THREAD_ROW = 4;
    const int ELE_PER_THREAD_COL = 4; // load (set 4 here because we use float4) and compute (equal to load number in squared block) ELE_PER_THREAD_COL elements in x dimension
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / ELE_PER_THREAD_COL, BLOCK_SIZE / ELE_PER_THREAD_ROW);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel_optim6_1_1<BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}