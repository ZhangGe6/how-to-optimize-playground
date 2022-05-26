#include "params.h"
#include "utils.h"
#include "MMult.h"

// use wider datatype like float4 
// How can it speedup: https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4

// use float4 (consecutive) elements in x dimension based on gemm_kernel_optim3_1
// https://github1s.com/Cjkkkk/CUDA_gemm/blob/HEAD/src/cuda/dense_legacy.cu#L265
// fairly speedup

template <int BLOCK_SIZE> 
__global__ void gemm_kernel_optim5_1(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Each thread computes 4 element of Csub
    // by accumulating results into C_value
    float4 C_value = {0, 0, 0, 0};
    
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
            
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {

        // Get sub-(block)-matrix Asub (upper-left corner) of A
        float *Asub = A + blockIdx.y * (BLOCK_SIZE * K) + tile_idx;  
        // Get sub-(block)-matrix Bsub (upper-left corner) of B
        float *Bsub = B + tile_idx * N + blockIdx.x * BLOCK_SIZE;  

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads 4 element of each sub-matrix (across 4 consecutive colomns)
        FETCH_FLOAT4(A_shared[thread_row][4 * thread_col]) = FETCH_FLOAT4(Asub[OFFSET(thread_row, 4 * thread_col, K)]);
        FETCH_FLOAT4(B_shared[thread_row][4 * thread_col]) = FETCH_FLOAT4(Bsub[OFFSET(thread_row, 4 * thread_col, N)]);

        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        // Each thread calculate 4 element of each sub-matrix (1x4 consecutive colomns)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value.x += A_shared[thread_row][i] * B_shared[i][4 * thread_col];
            C_value.y += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 1];
            C_value.z += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 2];
            C_value.w += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 3];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    C[OFFSET(row, 4 * col, N)] = C_value.x;
    C[OFFSET(row, 4 * col + 1, N)] = C_value.y;
    C[OFFSET(row, 4 * col + 2, N)] = C_value.z;
    C[OFFSET(row, 4 * col + 3, N)] = C_value.w;
}

void MMult_optim5_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE);   // decrease the thread dim along x dim by 4
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel_optim5_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
}

// use float4 based on gemm_kernel_optim3_2
// a large speedup
template <int BLOCK_SIZE, int ELE_PER_THREAD_ROW, int ELE_PER_THREAD_COL> 
__global__ void gemm_kernel_optim5_2(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // float4 C_value[ELE_PER_THREAD_ROW] = {{0, 0, 0, 0}};
    // ELE_PER_THREAD_COL == 4 here
    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {{0, 0, 0, 0}};

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {

        // Get sub-(block)-matrix Asub (upper-left corner) of A
        float *Asub = A + blockIdx.y * (BLOCK_SIZE * K) + tile_idx;  
        // Get sub-(block)-matrix Bsub (upper-left corner) of B
        float *Bsub = B + tile_idx * N + blockIdx.x * BLOCK_SIZE;  

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads (ELE_PER_THREAD_ROW * ELE_PER_THREAD_COL) `consective` element of each sub-matrix
        // :star: use thread.y, thread.x to map the location of shared memory
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
            FETCH_FLOAT4(A_shared[row_in_shared][4 * thread_col]) = FETCH_FLOAT4(Asub[OFFSET(row_in_shared, 4 * thread_col, K)]);
            FETCH_FLOAT4(B_shared[row_in_shared][4 * thread_col]) = FETCH_FLOAT4(Bsub[OFFSET(row_in_shared, 4 * thread_col, N)]);
        }
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // each thread is responsible for an `expanded` area
        // [thread_row * ELE_PER_THREAD_ROW:(thread_row + 1) * ELE_PER_THREAD_ROW][thread_col * ELE_PER_THREAD_COL:(thread_col + 1) * ELE_PER_THREAD_COL]
        
        // ========== v1 ========== //
        // #pragma unroll
        // for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
        //     #pragma unroll
        //     for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
        //         int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
        //         int col_in_shared = thread_col * ELE_PER_THREAD_COL + col_offset;
        //         #pragma unroll
        //         for (int i = 0; i < BLOCK_SIZE; ++i) {
        //             // register level writing
        //             C_value[row_offset][col_offset] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared];
        //         }
        //     }
        // }
        
        // ========== v2 ========== //
        // I find v2 is slightly faster (3860 vs. 3680 GFLOPs) than v1. Maybe we should explicitly unroll as many as possible? 
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
            int col_in_shared = thread_col * 4;
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                // register level writing
                // C_value[row_offset][col_offset] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared];
                C_value[row_offset][0] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared];
                C_value[row_offset][1] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared + 1];
                C_value[row_offset][2] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared + 2];
                C_value[row_offset][3] += A_shared[row_in_shared][i] * B_shared[i][col_in_shared + 3];
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

void MMult_optim5_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

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

    gemm_kernel_optim5_2<BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}