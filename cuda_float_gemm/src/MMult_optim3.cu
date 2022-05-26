#include "params.h"
#include "utils.h"
#include "MMult.h"

// increase the workload of each thread
// How can it speedup: https://cnugteren.github.io/tutorial/pages/page5.html

// each thread [load and compute] 4 (consecutive) elements in x dimension
// https://github1s.com/Cjkkkk/CUDA_gemm/blob/HEAD/src/cuda/dense_legacy.cu#L215-L216
// fairly speedup
template <int BLOCK_SIZE> 
__global__ void gemm_kernel_optim3_1(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Each thread computes 4 element of Csub
    // by accumulating results into C_value
    float C_value[4] = {0, 0, 0, 0};
    
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
            
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE) {

        // Get sub-(block)-matrix Asub (upper-left corner) of A
        float *Asub = A + blockIdx.y * (BLOCK_SIZE * K) + tile_idx;  
        // Get sub-(block)-matrix Bsub (upper-left corner) of B
        float *Bsub = B + tile_idx * N + blockIdx.x * BLOCK_SIZE;  

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads 4 element of each sub-matrix (across 4 consecutive colomns)
        A_shared[thread_row][4 * thread_col] = Asub[OFFSET(thread_row, 4 * thread_col, K)];
        A_shared[thread_row][4 * thread_col + 1] = Asub[OFFSET(thread_row, 4 * thread_col + 1, K)];
        A_shared[thread_row][4 * thread_col + 2] = Asub[OFFSET(thread_row, 4 * thread_col + 2, K)];
        A_shared[thread_row][4 * thread_col + 3] = Asub[OFFSET(thread_row, 4 * thread_col + 3, K)];

        B_shared[thread_row][4 * thread_col] = Bsub[OFFSET(thread_row, 4 * thread_col, N)];
        B_shared[thread_row][4 * thread_col + 1] = Bsub[OFFSET(thread_row, 4 * thread_col + 1, N)];
        B_shared[thread_row][4 * thread_col + 2] = Bsub[OFFSET(thread_row, 4 * thread_col + 2, N)];
        B_shared[thread_row][4 * thread_col + 3] = Bsub[OFFSET(thread_row, 4 * thread_col + 3, N)];
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        // Each thread calculate 4 element of each sub-matrix (1x4 consecutive colomns)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value[0] += A_shared[thread_row][i] * B_shared[i][4 * thread_col];
            C_value[1] += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 1];
            C_value[2] += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 2];
            C_value[3] += A_shared[thread_row][i] * B_shared[i][4 * thread_col + 3];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    C[OFFSET(row, 4 * col, N)] = C_value[0];
    C[OFFSET(row, 4 * col + 1, N)] = C_value[1];
    C[OFFSET(row, 4 * col + 2, N)] = C_value[2];
    C[OFFSET(row, 4 * col + 3, N)] = C_value[3];
}

void MMult_optim3_1(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE);   // decrease the thread dim along x dim by 4
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel_optim3_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
}

// each thread compute multiple (consecutive) elements in both x and y dimensions 
// a large speedup
// and `#pragma unroll` really matters here
// About how `#pragma unroll`work, see
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll
template <int BLOCK_SIZE, int ELE_PER_THREAD_ROW, int ELE_PER_THREAD_COL> 
__global__ void gemm_kernel_optim3_2(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {0};

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
            #pragma unroll
            for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
                int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
                int col_in_shared = thread_col * ELE_PER_THREAD_COL + col_offset;

                A_shared[row_in_shared][col_in_shared] = Asub[OFFSET(row_in_shared, col_in_shared, K)];
                B_shared[row_in_shared][col_in_shared] = Bsub[OFFSET(row_in_shared, col_in_shared, N)];
            }
        }
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // each thread is responsible for an `expanded` area
        // [thread_row * ELE_PER_THREAD_ROW:(thread_row + 1) * ELE_PER_THREAD_ROW][thread_col * ELE_PER_THREAD_COL:(thread_col + 1) * ELE_PER_THREAD_COL]
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            #pragma unroll
            for (int col_offset = 0; col_offset < ELE_PER_THREAD_COL; ++col_offset) {
                int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
                int col_in_shared = thread_col * ELE_PER_THREAD_COL + col_offset;
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; ++i) {
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
            
            C[OFFSET(row_in_C, col_in_C, N)] = C_value[row_offset][col_offset];   // d_C(row_in_C, col_in_C)
        }
    }

}

void MMult_optim3_2(cublasHandle_t handle, float *A, float *B, float *C, int M, int K, int N) {

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
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel_optim3_2<BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}