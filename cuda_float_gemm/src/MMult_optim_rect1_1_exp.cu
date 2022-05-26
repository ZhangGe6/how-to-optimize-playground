#include "params.h"
#include "utils.h"
#include "MMult.h"


// use relative indexing when loading data from global memory into shared memory
// ??? drop from 4800 to 3158
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect1_1_1( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N
    ) 
    {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    
    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // =================== exp here ===================> // 
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = A + blockIdx.y * (BLOCK_SIZE_M * K) + tile_idx;   // can only access blockIdx.y
        // // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = B + tile_idx * N + blockIdx.x * BLOCK_SIZE_N;   // can only access blockIdx.x
        
        // // load A_shared
        #pragma unroll
        for (int start_row = 0; start_row < BLOCK_SIZE_M; start_row += A_TILE_ROW_STRIDE) {
            int read_row = start_row + A_TILE_ROW_START;
            #pragma unroll
            for (int ele_offset = 0; ele_offset < 4; ++ele_offset) {
                int read_col = A_TILE_COL + ele_offset;
                As[read_row][read_col] = Asub[read_row * K + read_col];
            }
        }
        // load B_shared
        #pragma unroll
        for (int start_row = 0; start_row < BLOCK_SIZE_K; start_row += B_TILE_ROW_STRIDE) {
            int read_row = start_row + B_TILE_ROW_START;
            #pragma unroll
            for (int ele_offset = 0; ele_offset < 4; ++ele_offset) {
                int read_col = B_TILE_COL + ele_offset;
                Bs[read_row][read_col] = Bsub[read_row * N + read_col];
            }
        }
        // <=================== exp here =================== // 
    
        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                frag_b[thread_x] = Bs[k][THREAD_SIZE_X * tx + thread_x];
            }
            
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

void MMult_optim_rect1_1_1(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 128, 32, 128 ~ 4800 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 128, 16, 128 ~ 4400 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 32, 32, 32 ~ 2080 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64 ~  1100 GFLOPs

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect1_1_1<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

// not use register blocking
// ??? drop from 4800 to 2554
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect1_1_2( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N
    ) 
    {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    // float frag_a[THREAD_SIZE_Y];
    // float frag_b[THREAD_SIZE_X];
    
    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL + BLOCK_SIZE_N * bx, // col
                    N )]);
        }
    
        __syncthreads();

        // compute c
        // =================== exp here ===================> //
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            int row_in_shared = threadIdx.y * THREAD_SIZE_Y + thread_y;
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                int col_in_shared = threadIdx.x * THREAD_SIZE_X + thread_x;
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
                    // accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                    accum[thread_y][thread_x] += As[row_in_shared][k] * Bs[k][col_in_shared];
                }
            } 
        }
        // <=================== exp here =================== // 
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

void MMult_optim_rect1_1_2(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 128, 32, 128 ~ 4800 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 128, 16, 128 ~ 4400 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 32, 32, 32 ~ 2080 GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64 ~  1100 GFLOPs

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect1_1_2<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

