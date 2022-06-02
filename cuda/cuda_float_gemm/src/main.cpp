// #include <iostream>

#include "params.h"
#include "utils.h"
#include "MMult.h"


int main() {
    print_gpu_info();

    FILE *fptr;
    // fptr = fopen("../res/MMul_benchmark.txt","w");
    // fptr = fopen("../res/MMul_base.txt","w");
    // fptr = fopen("../res/MMul_optim5_2.txt", "w");
    fptr = fopen("../res/MMult_optim_rect2_4.txt", "w");
    // fptr = fopen("../res/MMult_optim_rect6_1_1.txt", "w");
    if(fptr == NULL)
    {
        printf("Error open result file!");   
        exit(1);             
    }

    for (int msize = 1024; msize <= 6144; msize += 128){ 
    // for (int msize = 4; msize <= 4; msize += 128){ 
        int M = msize, K = msize, N = msize;
        // int M = 1024, K = 16, N = 1024;

        /* each item of output require 2K floating point ops (multiply & add) and perform M*N times 
        See https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/ for more details*/
        float gflops = 2.0 * M * N * K * 1.0e-09;
        float time_best = DBL_MAX;
        // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
        cudaEvent_t start, stop;
        cudaEventCreate(&start);  cudaEventCreate(&stop);
        // https://docs.nvidia.com/cuda/cublas/index.html#cublashandle_t
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        // host side
        float *h_A, *h_B, *h_C_base, *h_C_optim;
        h_A = (float*) malloc(ASIZE(float));
        h_B = (float*) malloc(BSIZE(float));
        h_C_base = (float*) malloc(CSIZE(float));
        h_C_optim = (float*) malloc(CSIZE(float));

        // device side
        float *d_A, *d_B, *d_C;
        checkCudaErrors(cudaMalloc(&d_A, ASIZE(float)));
        checkCudaErrors(cudaMalloc(&d_B, BSIZE(float)));
        checkCudaErrors(cudaMalloc(&d_C, CSIZE(float)));
        
        // init data in the host side
        random_matrix(h_A, M, K);
        random_matrix(h_B, K, N);
        zero_matrix(h_C_base, M, N);
        zero_matrix(h_C_optim, M, N);

        // copy data from host to device
        checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(float), cudaMemcpyHostToDevice)); 
        checkCudaErrors(cudaMemcpy(d_C, h_C_base, CSIZE(float), cudaMemcpyHostToDevice));  
        checkCudaErrors(cudaMemcpy(d_C, h_C_optim, CSIZE(float), cudaMemcpyHostToDevice));  

        // cublas baseline for result verification
        MMult_benchmark(handle, d_A, d_B, d_C, M, K, N);
        checkCudaErrors(cudaMemcpy(h_C_base, d_C, CSIZE(float), cudaMemcpyDeviceToHost));
        // print_matrix(M, N, h_C_base, ldc);

        int repeat_times = 2;
        for (int repeat = 0; repeat < repeat_times; ++repeat) {
            zero_matrix(h_C_optim, M, N);  // because we are doing an [inplace] adding operation on C_optim, so we need to initialize C_optim every iter
            checkCudaErrors(cudaMemcpy(d_C, h_C_optim, CSIZE(float), cudaMemcpyHostToDevice));  
            cudaEventRecord(start);

            // MMult_benchmark(handle, d_A, d_B, d_C, M, K, N);
            // MMult_base(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim1_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim2_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim3_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim3_2(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim4_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim5_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim5_2(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim6_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim6_1_1(handle, d_A, d_B, d_C, M, K, N);
            
            // MMult_optim_rect1_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim_rect1_1_1(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim_rect1_1_2(handle, d_A, d_B, d_C, M, K, N);
            // MMult_optim_rect2_1(handle, d_A, d_B, d_C, M, K, N);
            MMult_optim_rect2_4(handle, d_A, d_B, d_C, M, K, N);
            


            cudaEventRecord(stop);

            checkCudaErrors(cudaMemcpy(h_C_optim, d_C, CSIZE(float), cudaMemcpyDeviceToHost));
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            time_best = MIN(time_best, milliseconds / 1000);
        }
        // print_matrix(h_A, M, K);
        // print_matrix(h_B, K, N);
        // print_matrix(h_C_base, M, N);
        // print_matrix(h_C_optim, M, N);
        // printf("time best %f\n", time_best);
        // print_matrix(M, N, C_optim, ldc);
        // print_matrix(M, N, C_base, ldc);

        float max_diff = compare_matrix(h_C_base, h_C_optim, M, N);
        // assert(max_diff == 0);
        // printf("max diff %f\n", max_diff);
        // assert(max_diff < 0.000001);
        printf( "%d %f %f \n", msize, gflops / time_best, max_diff);
        fprintf(fptr,"%d %f %f \n", msize, gflops / time_best, max_diff);

        // free memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(h_A);
        free(h_B);
        free(h_C_base);
        free(h_C_optim);
    }
    fclose(fptr); 
}