// #include <iostream>

#include "params.h"
#include "utils.h"
#include "MMult.h"

int main() {
    FILE *fptr;
    // fptr = fopen("../res/MMul_base.txt","w");
    fptr = fopen("../res/MMul_optim9_1.txt", "w");
    if(fptr == NULL)
    {
        printf("Error open result file!");   
        exit(1);             
    }

    // for (int msize = 40; msize <= 800; msize += 40){   
    for (int msize = 4; msize <= 4; msize += 40){  
        int m = msize, k = msize, n = msize;
        int lda = k, ldb = n, ldc = n;

        /* each item of output require 2K floating point ops (multiply & add) and perform M*K times 
        See https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/ for more details*/
        float gflops = 2.0 * m * n * k * 1.0e-09;
        float time_best = DBL_MAX;
        
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
        random_matrix(m, k, h_A, lda);
        random_matrix(k, n, h_B, ldb);
        zero_matrix(m, n, h_C_base, ldc);
        zero_matrix(m, n, h_C_optim, ldc);

        // copy data from host to device
        checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(float), cudaMemcpyHostToDevice)); 

        checkCudaErrors(cudaMemcpy(d_C, h_C_base, CSIZE(float), cudaMemcpyHostToDevice));   
        // cublas baseline for result verification
        MMult_benchmark(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);
        checkCudaErrors(cudaMemcpy(h_C_base, d_C, CSIZE(float), cudaMemcpyDeviceToHost));
        print_matrix(m, n, h_C_base, ldc);

        int repeat_times = 5;   // TODO: when repeat_times is lager than 1, the max_diff is wierd. -> I know, 
        for (int repeat = 0; repeat < repeat_times; ++repeat) {
            zero_matrix(m, n, h_C_optim, ldc);  // because we are doing an [inplace] adding operation on C_optim, so we need to initialize C_optim every iter
            checkCudaErrors(cudaMemcpy(d_C, h_C_optim, CSIZE(float), cudaMemcpyHostToDevice));  
            float time_s = dclock();

            MMult_benchmark(m, k, n, d_A, d_B, d_C, lda, ldb, ldc);

            checkCudaErrors(cudaMemcpy(h_C_optim, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

            
            // time_best = MIN(time_best, (dclock() - time_s));
        }
        // print_matrix(m, n, C_optim, ldc);
        // print_matrix(m, n, C_base, ldc);

        float max_diff = compare_matrix(m, n, h_C_base, h_C_optim, ldc);
        assert(max_diff == 0);
        // printf( "%d %f %f \n", msize, gflops / time_best, max_diff);
        // fprintf(fptr,"%d %f %f \n", msize, gflops / time_best, max_diff);

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