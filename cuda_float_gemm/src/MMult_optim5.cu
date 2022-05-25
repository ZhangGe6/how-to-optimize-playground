#include "params.h"
#include "utils.h"
#include "MMult.h"

// use more flexible block size for m, n and k based on MMult_optim2_1
template<
    int BLOCK_SIZE_M, 
    int BLOCK_SIZE_N, 
    int BLOCK_SIZE_K
> 
__global__ void gemm_optim5_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    // Each thread computes one element of Csub
    // by accumulating results into C_value
    float C_value = 0;

    __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // 各线程依次完成“千层饼”的第tile_k_id * BLOCK_SIZE~ tile_k_id * BLOCK_SIZE + BLOCK_SIZE层，并叠加
    // printf("k %d, BLOCK_SIZE %d\n", k, BLOCK_SIZE);
    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE_K); ++tile_k_id) {
        // printf("tile_k_id %d\n", tile_k_id);
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE_M * k) + tile_k_id * BLOCK_SIZE_K;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE_K * n) + blockIdx.x * BLOCK_SIZE_N;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // TODO: I do not know how to load data without branching when the threads per block is more than the elements in the shared memory (aka BLOCK_SIZE_K < BLOCK_SIZE_N or BLOCK_SIZE_K < BLOCK_SIZE_M)
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        A_shared[row_in_block][col_in_block] = Asub[row_in_block * lda + col_in_block];
        B_shared[row_in_block][col_in_block] = Bsub[row_in_block * ldb + col_in_block];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        for (int i = 0; i < BLOCK_SIZE_K; ++i) {
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

void MMult_optim5_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // const int BLOCK_SIZE_M = 128;
    // const int BLOCK_SIZE_N = 128;
    // const int BLOCK_SIZE_K = 32;
    
    printf("I do not know how to load data without branching when the threads per block is more than the elements in the shared memory (aka BLOCK_SIZE_K < BLOCK_SIZE_N or BLOCK_SIZE_K < BLOCK_SIZE_M), so this one is not implemented");
    assert(false);

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 16;

    dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 dimGrid((n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_optim5_1<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K> <<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


// use more flexible block size for m, n and k based on MMult_optim3_2
template<
    int BLOCK_SIZE_M, 
    int BLOCK_SIZE_N, 
    int BLOCK_SIZE_K, 
    int ELE_PER_THREAD_ROW, 
    int ELE_PER_THREAD_COL,
    int ELE_PER_THREAD_READ
> 
__global__ void gemm_optim5_2(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;
    
    int thread_row = threadIdx.y, thread_col = threadIdx.x;

    // by accumulating results into C_value
    float C_value[ELE_PER_THREAD_ROW][ELE_PER_THREAD_COL] = {0};

    const int block_thread_num = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int A_thread_num_read_per_row = BLOCK_SIZE_K / ELE_PER_THREAD_READ;   // the thread num needed to read a row
    const int A_row_stride = block_thread_num / A_thread_num_read_per_row;      // after all threads finish reading, we move down for next sub-block
    
    int A_read_row_offset = tid / A_thread_num_read_per_row;
    int A_read_col_start = (tid % A_thread_num_read_per_row) * ELE_PER_THREAD_READ;

    const int B_thread_num_read_per_row = BLOCK_SIZE_N / ELE_PER_THREAD_READ;   // the thread num needed to read a row
    const int B_row_stride = block_thread_num / B_thread_num_read_per_row;      // after all threads finish reading, we move down for next sub-block
    
    int B_read_row_offset = tid / B_thread_num_read_per_row;
    int B_read_col_start = (tid % B_thread_num_read_per_row) * ELE_PER_THREAD_READ;

    __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE_K); ++tile_k_id) {

        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE_M * k) + tile_k_id * BLOCK_SIZE_K;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE_K * n) + blockIdx.x * BLOCK_SIZE_N;   // can only access blockIdx.x

        // load A_shared
        #pragma unroll
        for (int start_row = 0; start_row < BLOCK_SIZE_M; start_row += A_row_stride) {
            int read_row = start_row + A_read_row_offset;
            #pragma unroll
            for (int ele_offset = 0; ele_offset < ELE_PER_THREAD_READ; ++ele_offset) {
                int read_col = A_read_col_start + ele_offset;
                A_shared[read_row][read_col] = Asub[read_row * lda + read_col];
            }
        }
        
        // load B_shared
        #pragma unroll
        for (int start_row = 0; start_row < BLOCK_SIZE_K; start_row += B_row_stride) {
            int read_row = start_row + B_read_row_offset;
            #pragma unroll
            for (int ele_offset = 0; ele_offset < ELE_PER_THREAD_READ; ++ele_offset) {
                int read_col = B_read_col_start + ele_offset;
                B_shared[read_row][read_col] = Bsub[read_row * ldb + read_col];
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
                int row_in_shared = thread_row * ELE_PER_THREAD_ROW + row_offset;
                int col_in_shared = thread_col * ELE_PER_THREAD_COL + col_offset;
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

// 疑问：为什么BLOCK_SIZE_K = 32正常跑，BLOCK_SIZE_K=8反而报“GPUassert: too many resources requested for launch”？
void MMult_optim5_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 32;
    const int ELE_PER_THREAD_ROW = 4;
    const int ELE_PER_THREAD_COL = 4;
    const int ELE_PER_THREAD_READ = 4;

    dim3 dimBlock(BLOCK_SIZE_N / ELE_PER_THREAD_COL, BLOCK_SIZE_M / ELE_PER_THREAD_ROW);
    dim3 dimGrid((m + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (n + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_optim5_2<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL, ELE_PER_THREAD_READ> <<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
