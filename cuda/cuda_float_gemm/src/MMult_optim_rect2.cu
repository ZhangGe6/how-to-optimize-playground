#include "params.h"
#include "utils.h"
#include "MMult.h"

// the base version from https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemm/sgemm_v1.cu
template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect2_0( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
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

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

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

void MMult_optim_rect2_0(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {


    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect2_0<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

// ======= The following versions are written by myself from step to step ======= // 

// enable double buffer
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect2_1( 
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


    // ======= base version =======> // 
    // __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    // __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // // can not unroll since K can not be determined at this point
    // for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
    //     // load A from global memory to shared memory
    //     #pragma unroll
    //     for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
    //         FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
    //                 BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
    //                 A_TILE_COL + tile_idx, // col
    //                 K )]);
    //     }

    //     // load B from global memory to shared memory
    //     #pragma unroll
    //     for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    //         FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
    //                 tile_idx + B_TILE_ROW_START + i, // row
    //                 B_TILE_COL + BLOCK_SIZE_N * bx, // col
    //                 N )]);
    //     }

    //     __syncthreads();

    //     // compute c
    //     #pragma unroll
    //     for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
    //         // load A from shared memory to register
    //         #pragma unroll
    //         for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
    //             frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
    //         }

    //         // load B from shared memory to register
    //         #pragma unroll
    //         for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
    //             frag_b[thread_x] = Bs[k][THREAD_SIZE_X * tx + thread_x];
    //         }

    //         #pragma unroll
    //         for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
    //             #pragma unroll
    //             for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
    //                 accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
    //             }
    //         }

    //     }
    //     __syncthreads();
    // }
    // <======= base version ======= // 

    // ======= prefetch version =======> // 
    // ======= prefetch for the 1st tile =======> // 
    __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL/*+0*/, // col
                K )]);
    }

    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                /*0+*/B_TILE_ROW_START + i, // row
                B_TILE_COL + BLOCK_SIZE_N * bx, // col
                N )]);
    }
    // <======= prefetch for the 1st tile ======= // 

    int load_stage_idx = 0;                             // can load from load_stage_idx
    int write_stage_idx = load_stage_idx ^ 1; // XNOR   // can write to write_stage_idx
    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {  // computing iter
        // load A from global memory to shared memory
        if (tile_idx + BLOCK_SIZE_K < K) {
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                        BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                        A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                        K )]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                        (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                        B_TILE_COL + BLOCK_SIZE_N * bx, // col
                        N )]);
            }
            __syncthreads();
        }

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[load_stage_idx][ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                frag_b[thread_x] = Bs[load_stage_idx][k][THREAD_SIZE_X * tx + thread_x];
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

        write_stage_idx = load_stage_idx;
        load_stage_idx = load_stage_idx ^ 1;
    }
    // ======= prefetch version =======> // 

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

void MMult_optim_rect2_1(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    // const int ELE_PER_THREAD_READ = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect2_0<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

// split transfer of B
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect2_2( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N) {
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

    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    // about 'ldg': https://forums.developer.nvidia.com/t/whats-different-between-ld-and-ldg-load-from-generic-memory-vs-load-from-global-memory/40856/3
    // const int ldg_a_per_thread = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
    const int ldg_b_per_thread = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;
    // float ldg_a_reg[ldg_a_per_thread];
    float ldg_b_reg[ldg_b_per_thread];

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

    // ======= prefetch version =======> // 
    // ======= prefetch for the 1st tile =======> // 
    __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL/*+0*/, // col
                K )]);
    }

    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        // FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
        //         /*0+*/B_TILE_ROW_START + i, // row
        //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
        //         N )]);

        // move from global memory to tmp register
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;   // the start index of each segment
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
            /*0+*/i + B_TILE_ROW_START, // row
            BLOCK_SIZE_N * bx + B_TILE_COL, // col
            N )]);
        // move from tmp register to shared memory
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
    }
    // <======= prefetch for the 1st tile ======= // 

    int load_stage_idx = 0;                             // can load from load_stage_idx
    int write_stage_idx = load_stage_idx ^ 1; // XNOR   // can write to write_stage_idx
    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {  // computing iter
        // load A from global memory to shared memory
        if (tile_idx + BLOCK_SIZE_K < K) {   // TODO: ??????branching??????????????????
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                        BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                        A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                        K )]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                        (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                        B_TILE_COL + BLOCK_SIZE_N * bx, // col
                        N )]);
            }
            __syncthreads();
        }


        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[load_stage_idx][ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                frag_b[thread_x] = Bs[load_stage_idx][k][THREAD_SIZE_X * tx + thread_x];
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

        write_stage_idx = load_stage_idx;
        load_stage_idx = load_stage_idx ^ 1;
    }
    // ======= prefetch version =======> // 

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

void MMult_optim_rect2_2(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    // const int ELE_PER_THREAD_READ = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect2_2<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}


// split transfer of A and B
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect2_3( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N) {
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

    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    // about 'ldg': https://forums.developer.nvidia.com/t/whats-different-between-ld-and-ldg-load-from-generic-memory-vs-load-from-global-memory/40856/3
    const int ldg_a_per_thread = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
    const int ldg_b_per_thread = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;
    float ldg_a_reg[ldg_a_per_thread];
    float ldg_b_reg[ldg_b_per_thread];

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

    // ======= prefetch version =======> // 
    // ======= prefetch for the 1st tile =======> // 
    __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K]; 
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        // FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
        //         BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
        //         A_TILE_COL/*+0*/, // col
        //         K )]);
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;   // the start index of each segment
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL/*+0*/, // col
                K )]);
        FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(ldg_a_reg[ldg_index]);
    }

    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        // FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
        //         /*0+*/B_TILE_ROW_START + i, // row
        //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
        //         N )]);

        // move from global memory to tmp register
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;   // the start index of each segment
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
            /*0+*/i + B_TILE_ROW_START, // row
            BLOCK_SIZE_N * bx + B_TILE_COL, // col
            N )]);
        // move from tmp register to shared memory
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
    }
    // <======= prefetch for the 1st tile ======= // 

    int load_stage_idx = 0;                             // can load from load_stage_idx
    int write_stage_idx = load_stage_idx ^ 1; // XNOR   // can write to write_stage_idx
    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {  // computing iter
        // load A from global memory to shared memory
        if (tile_idx + BLOCK_SIZE_K < K) {   // TODO: ??????branching??????????????????
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                //         BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                //         A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                //         K )]);
                int ldg_index = i / A_TILE_ROW_STRIDE * 4; 
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                    K )]);
                FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(ldg_a_reg[ldg_index]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                //         (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
                //         N )]);
                int ldg_index = i / B_TILE_ROW_STRIDE * 4; 
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                        (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                        B_TILE_COL + BLOCK_SIZE_N * bx, // col
                        N )]);
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
        }


        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[load_stage_idx][ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                frag_b[thread_x] = Bs[load_stage_idx][k][THREAD_SIZE_X * tx + thread_x];
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

        write_stage_idx = load_stage_idx;
        load_stage_idx = load_stage_idx ^ 1;
    }
    // ======= prefetch version =======> // 

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

void MMult_optim_rect2_3(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    // const int ELE_PER_THREAD_READ = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect2_3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

// scatter the transfer-splited A and B
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm_kernel_optim_rect2_4( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N) {
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

    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    // about 'ldg': https://forums.developer.nvidia.com/t/whats-different-between-ld-and-ldg-load-from-generic-memory-vs-load-from-global-memory/40856/3
    const int ldg_a_per_thread = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
    const int ldg_b_per_thread = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;
    float ldg_a_reg[ldg_a_per_thread];
    float ldg_b_reg[ldg_b_per_thread];

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

    // ======= prefetch version =======> // 
    // ======= prefetch for the 1st tile =======> // 
    __shared__ float As[2][BLOCK_SIZE_M][BLOCK_SIZE_K]; 
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        // FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
        //         BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
        //         A_TILE_COL/*+0*/, // col
        //         K )]);
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;   // the start index of each segment
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL/*+0*/, // col
                K )]);
        FETCH_FLOAT4(As[0][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(ldg_a_reg[ldg_index]);
    }

    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        // FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
        //         /*0+*/B_TILE_ROW_START + i, // row
        //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
        //         N )]);

        // move from global memory to tmp register
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;   // the start index of each segment
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
            /*0+*/i + B_TILE_ROW_START, // row
            BLOCK_SIZE_N * bx + B_TILE_COL, // col
            N )]);
        // move from tmp register to shared memory
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
    }
    // <======= prefetch for the 1st tile ======= // 

    int load_stage_idx = 0;                             // can load from load_stage_idx
    int write_stage_idx = load_stage_idx ^ 1; // XNOR   // can write to write_stage_idx
    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {  // computing iter
        // load A from global memory to shared memory
        if (tile_idx + BLOCK_SIZE_K < K) {   // TODO: ??????branching??????????????????
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                //         BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                //         A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                //         K )]);
                int ldg_index = i / A_TILE_ROW_STRIDE * 4; 
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                    K )]);
                // FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(ldg_a_reg[ldg_index]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                //         (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
                //         N )]);
                int ldg_index = i / B_TILE_ROW_STRIDE * 4; 
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                        (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                        B_TILE_COL + BLOCK_SIZE_N * bx, // col
                        N )]);
                // FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
        }

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[load_stage_idx][ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                frag_b[thread_x] = Bs[load_stage_idx][k][THREAD_SIZE_X * tx + thread_x];
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

        if (tile_idx + BLOCK_SIZE_K < K) {   // TODO: ??????branching??????????????????
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                //         BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                //         A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                //         K )]);
                int ldg_index = i / A_TILE_ROW_STRIDE * 4; 
                // FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                //     BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                //     A_TILE_COL + (tile_idx + BLOCK_SIZE_K), // col
                //     K )]);
                FETCH_FLOAT4(As[write_stage_idx][A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(ldg_a_reg[ldg_index]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                // FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                //         (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
                //         N )]);
                int ldg_index = i / B_TILE_ROW_STRIDE * 4; 
                // FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                //         (tile_idx + BLOCK_SIZE_K) + B_TILE_ROW_START + i, // row
                //         B_TILE_COL + BLOCK_SIZE_N * bx, // col
                //         N )]);
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
        }

        write_stage_idx = load_stage_idx;
        load_stage_idx = load_stage_idx ^ 1;
    }
    // ======= prefetch version =======> // 

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

void MMult_optim_rect2_4(cublasHandle_t handle, float *A, float *B, float *C, const int M, const int K, const int N) {

    // params really matters:
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 64, 64, 64, 4, 4 ~ 2450GFLOPs
    // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL = 

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    // const int ELE_PER_THREAD_READ = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid((M + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    gemm_kernel_optim_rect2_4<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> <<<dimGrid, dimBlock>>>(A, B, C, M, K, N);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
} 