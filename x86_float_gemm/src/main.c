#include "params.h"
#include "utils.h"
#include "MMult.h"

int main() {
    FILE *fptr;
    // fptr = fopen("../res/MMult_base.txt","w");
    fptr = fopen("../res/MMult_optim7_3.txt", "w");
    if(fptr == NULL)
    {
        printf("Error open file!");   
        exit(1);             
    }

    for (int msize = 40; msize <= 1200; msize += 40){
    // for (int msize = 32; msize <= 1024; msize += 32){
    // for (int msize = 8; msize <= 8; msize += 4){   
    // for (int msize = 360; msize <= 360; msize += 4){    
        float *A, *B, *C_base, *C_optim;
        int M = msize, K = msize, N = msize;
        int lda = K, ldb = N, ldc = N;

        /* each item of output require 2K floating point ops (multiply & add) and perform M*K times 
        See https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/ for more details*/
        float gflops = 2.0 * M * N * K * 1.0e-09;
        float time_best = DBL_MAX;

        A = (float*) malloc(M * K * sizeof(float));
        B = (float*) malloc(K * N * sizeof(float));
        C_base = (float*) malloc(M * N * sizeof(float));
        C_optim = (float*) malloc(M * N * sizeof(float));
        random_matrix(A, M, K, lda);
        random_matrix(B, K, N, ldb);
        zero_matrix(C_base, M, N, ldc);
        zero_matrix(C_optim, M, N, ldc);
        
        MMult_base(A, B, C_base, M, K, N, lda, ldb, ldc);

        // printf("testing msize %d\N", msize);
        int repeat_times = 1; 
        for (int repeat = 0; repeat < repeat_times; ++repeat) {
            zero_matrix(C_optim, M, N, ldc);  // because we are doing an [inplace] adding operation on C_optim, so we need to initialize C_optim every iter
            // print_matrix(C_optim, M, N, ldc);
            float start = clock();

            // MMult_base(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim1_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim1_2(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim2_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim3_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim3_2(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim3_3(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim3_4(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim3_5(A, B, C_optim, M, K, N, lda, ldb, ldc);

            // MMult_optim4_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim4_2(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim4_3(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim4_4(A, B, C_optim, M, K, N, lda, ldb, ldc);

            // MMult_optim5_1(A, B, C_optim, M, K, N, lda, ldb, ldc);

            // MMult_optim6_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim6_2(A, B, C_optim, M, K, N, lda, ldb, ldc);

            // MMult_optim7_1(A, B, C_optim, M, K, N, lda, ldb, ldc);
            // MMult_optim7_2(A, B, C_optim, M, K, N, lda, ldb, ldc);
            MMult_optim7_3(A, B, C_optim, M, K, N, lda, ldb, ldc);





            // https://stackoverflow.com/a/459704/10096987
            float elapsed_seconds = (clock() - start) / CLOCKS_PER_SEC;
            time_best = MIN(time_best, elapsed_seconds);
        }
        // print_matrix(A, M, K, lda);
        // print_matrix(B, K, N, ldb);
        // print_matrix(C_base, M, N, ldc);
        // print_matrix(C_optim, M, N, ldc);
        // printf("max value in C_base %f\n", max_value(C_base, M, N, ldc));
        // printf("max value in C_optim %f\n", max_value(C_optim, M, N, ldc));

        // printf("time best %f\n", time_best);

        float max_diff = compare_matrix(C_base, C_optim, M, N, ldc);
        // assert(max_diff == 0);
        printf( "%d %f %f \n", msize, gflops / time_best, max_diff);
        fprintf(fptr,"%d %f %f \n", msize, gflops / time_best, max_diff);

        free(A);
        free(B);
        free(C_base);
        free(C_optim);
    }
    fclose(fptr); 
}