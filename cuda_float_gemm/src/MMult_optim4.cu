#include "params.h"
#include "utils.h"
#include "MMult.h"

// use wider datatype like float4 
// How can it speedup: https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4

// use float4 (consecutive) elements in x dimension based on gemm_optim3_1
// https://github1s.com/Cjkkkk/CUDA_gemm/blob/HEAD/src/cuda/dense_legacy.cu#L265
// fairly speedup
template <int BLOCK_SIZE> 
__global__ void gemm_optim4_1(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;

    // Each thread computes 4 element of Csub
    // by accumulating results into C_value
    float4 C_value = {0, 0, 0, 0};
    
    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {
        // __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
        // __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];
                
        // Get sub-(block)-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE * k) + tile_k_id * BLOCK_SIZE;  
        // Get sub-(block)-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE * n) + blockIdx.x * BLOCK_SIZE;  

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads 4 element of each sub-matrix (across 4 consecutive colomns)
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        // A_shared[row_in_block][4 * col_in_block] = Asub[row_in_block * lda + 4 * col_in_block];
        // A_shared[row_in_block][4 * col_in_block + 1] = Asub[row_in_block * lda + 4 * col_in_block + 1];
        // A_shared[row_in_block][4 * col_in_block + 2] = Asub[row_in_block * lda + 4 * col_in_block + 2];
        // A_shared[row_in_block][4 * col_in_block + 3] = Asub[row_in_block * lda + 4 * col_in_block + 3];
        // the amount of loads per thread may be reduced
        reinterpret_cast<float4*>(A_shared + row_in_block * BLOCK_SIZE + 4 * col_in_block)[0] = 
                                                    reinterpret_cast<float4*>(Asub + row_in_block * lda + 4 * col_in_block)[0];

        // B_shared[row_in_block][4 * col_in_block] = Bsub[row_in_block * ldb + 4 * col_in_block];
        // B_shared[row_in_block][4 * col_in_block + 1] = Bsub[row_in_block * ldb + 4 * col_in_block + 1];
        // B_shared[row_in_block][4 * col_in_block + 2] = Bsub[row_in_block * ldb + 4 * col_in_block + 2];
        // B_shared[row_in_block][4 * col_in_block + 3] = Bsub[row_in_block * ldb + 4 * col_in_block + 3];
        reinterpret_cast<float4*>(B_shared + row_in_block * BLOCK_SIZE + 4 * col_in_block)[0] = 
                                                    reinterpret_cast<float4*>(Bsub + row_in_block * ldb + 4 * col_in_block)[0];
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply within Asub and Bsub
        // Each thread calculate 4 element of each sub-matrix (1x4 consecutive colomns)
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            // C_value[0] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block];
            // C_value[1] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 1];
            // C_value[2] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 2];
            // C_value[3] += A_shared[row_in_block][i] * B_shared[i][4 * col_in_block + 3];
            C_value.x += A_shared[row_in_block * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block];
            C_value.y += A_shared[row_in_block * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 1];
            C_value.z += A_shared[row_in_block * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 2];
            C_value.w += A_shared[row_in_block * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 3];
        } 

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // TODO: without this, the results are calculated more efficiently and still correctly
        __syncthreads();  
    }

    // d_C(row, 4 * col) = C_value[0];
    // d_C(row, 4 * col + 1) = C_value[1];
    // d_C(row, 4 * col + 2) = C_value[2];
    // d_C(row, 4 * col + 3) = C_value[3];
    reinterpret_cast<float4*>(d_C + row * ldc + 4 * col)[0] = C_value;
}

void MMult_optim4_1(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE);   // decrease the thread dim along x dim by 4 (// Note that we use float4 in this version, so we set 4 and do not change it to other value)
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_optim4_1<BLOCK_SIZE><<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
}

// use float4 (consecutive) elements in both x and y dimension
// ELE_PER_THREAD_COL should be 4 in this kernel
// excellent speedup, too!
template <int BLOCK_SIZE, int ELE_PER_THREAD_ROW, int ELE_PER_THREAD_COL> 
__global__ void gemm_optim4_2(int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Note `row` and `col` are the index of threads, they are not on pair with the matrix element index in this version
    // if (row >= m || col >= n) return;

    // by accumulating results into C_value
    float4 C_value[ELE_PER_THREAD_ROW] = {{0, 0, 0, 0}};

    for (int tile_k_id = 0; tile_k_id < int(k / BLOCK_SIZE); ++tile_k_id) {
        __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];
        
        // Get sub-matrix Asub (upper-left corner) of A
        float *Asub = d_A + blockIdx.y * (BLOCK_SIZE * k) + tile_k_id * BLOCK_SIZE;   // can only access blockIdx.y
        // Get sub-matrix Bsub (upper-left corner) of B
        float *Bsub = d_B + tile_k_id * (BLOCK_SIZE * n) + blockIdx.x * BLOCK_SIZE;   // can only access blockIdx.x

        // Load Asub and Bsub from device memory to shared memory
        // :star: use thread.y, thread.x to map the location of shared memory
        int row_in_block = threadIdx.y, col_in_block = threadIdx.x;
        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            int row_in_shared = row_in_block * ELE_PER_THREAD_ROW + row_offset;
            reinterpret_cast<float4*>(A_shared + row_in_shared * BLOCK_SIZE + 4 * col_in_block)[0] = 
                                                reinterpret_cast<float4*>(Asub + row_in_shared * lda + 4 * col_in_block)[0]; 
            reinterpret_cast<float4*>(B_shared + row_in_shared * BLOCK_SIZE + 4 * col_in_block)[0] = 
                                                reinterpret_cast<float4*>(Bsub + row_in_shared * ldb + 4 * col_in_block)[0]; 
        }
        // ======== load done ======= //

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        #pragma unroll
        for (int row_offset = 0; row_offset < ELE_PER_THREAD_ROW; ++row_offset) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                C_value[row_offset].x += A_shared[(ELE_PER_THREAD_ROW * row_in_block + row_offset) * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block];
                C_value[row_offset].y += A_shared[(ELE_PER_THREAD_ROW * row_in_block + row_offset) * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 1];
                C_value[row_offset].z += A_shared[(ELE_PER_THREAD_ROW * row_in_block + row_offset) * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 2];
                C_value[row_offset].w += A_shared[(ELE_PER_THREAD_ROW * row_in_block + row_offset) * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + 4 * col_in_block + 3];
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
        reinterpret_cast<float4*>(d_C + (ELE_PER_THREAD_ROW * row + row_offset) * ldc + 4 * col)[0] = C_value[row_offset]; 
    }

}

void MMult_optim4_2(cublasHandle_t handle, int m, int k, int n, float *d_A, float *d_B, float *d_C, int lda, int ldb, int ldc) {

    // const int BLOCK_SIZE = 16;
    const int BLOCK_SIZE = 64;   // BLOCK_SIZE matters
    const int ELE_PER_THREAD_ROW = 4;  
    const int ELE_PER_THREAD_COL = 4;  // Note that we use float4 in this version, so we set 4 and do not change it to other value
    // const int BLOCK_SIZE = 128;   // error: uses too much shared data 
    dim3 dimBlock(BLOCK_SIZE / ELE_PER_THREAD_COL, BLOCK_SIZE / ELE_PER_THREAD_ROW);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_optim4_2<BLOCK_SIZE, ELE_PER_THREAD_ROW, ELE_PER_THREAD_COL> <<<dimGrid, dimBlock>>>(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}


