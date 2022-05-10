// #include <iostream>
#include "params.h"
#include "utils.h"
#include "MMult.h"

int main() {
    FILE *fptr;
    // fptr = fopen("../res/MMul_base.txt","w");
    fptr = fopen("../res/MMul_optim8_5.txt", "w");
    if(fptr == NULL)
    {
        printf("Error!");   
        exit(1);             
    }

    for (int msize = 40; msize <= 800; msize += 40){
    // for (int msize = 8; msize <= 8; msize += 4){    
        double *A, *B, *C_base, *C_optim;
        int m = msize, k = msize, n = msize;
        int lda = m, ldb = k, ldc = m;

        /* each item of output require 2K floating point ops (multiply & add) and perform M*K times 
        See https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/ for more details*/
        double gflops = 2.0 * m * n * k * 1.0e-09;
        double time_best = DBL_MAX;

        A = (double*) malloc(m * k * sizeof(double));
        B = (double*) malloc(k * n * sizeof(double));
        C_base = (double*) malloc(m * n * sizeof(double));
        C_optim = (double*) malloc(m * n * sizeof(double));
        random_matrix(m, k, A, lda);
        random_matrix(k, n, B, ldb);
        zero_matrix(m, n, C_base, ldc);
        zero_matrix(m, n, C_optim, ldc);
        
        MMult_base(m, k, n, A, B, C_base, lda, ldb, ldc);
        // print_matrix(m, k, A, lda);
        // print_matrix(k, n, B, ldb);
        // print_matrix(m, n, C, ldc);

        // printf("testing msize %d\n", msize);
        int repeat_times = 5;   // TODO: when repeat_times is lager than 1, the max_diff is wierd. -> I know, 
        for (int repeat = 0; repeat < repeat_times; ++repeat) {
            zero_matrix(m, n, C_optim, ldc);  // because we are doing an [inplace] adding operation on C_optim, so we need to initialize C_optim every iter
            double time_s = dclock();

            // MMult_base(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim1_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim1_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim1_3(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim1_4(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim2_0(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim2_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim2_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim2_3(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim3_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_3(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_4(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_5(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_6(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim3_7(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim4_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim4_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim4_3(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim4_4(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim4_5(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim5_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim5_2(m, k, n, A, B, C_optim, lda, ldb, ldc); 

            // MMult_optim6_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_3(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_4(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_5(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_6(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim6_7(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim7_1(m, k, n, A, B, C_optim, lda, ldb, ldc);

            // MMult_optim8_0(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_1(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_2(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_3(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_4(m, k, n, A, B, C_optim, lda, ldb, ldc);
            MMult_optim8_5(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_6(m, k, n, A, B, C_optim, lda, ldb, ldc);
            // MMult_optim8_7(m, k, n, A, B, C_optim, lda, ldb, ldc);


            time_best = MIN(time_best, (dclock() - time_s));
        }
        // print_matrix(m, n, C_optim, ldc);
        // print_matrix(m, n, C_base, ldc);

        double max_diff = compare_matrix(m, n, C_base, C_optim, ldc);
        assert(max_diff == 0);
        printf( "%d %f %f \n", msize, gflops / time_best, max_diff);
        fprintf(fptr,"%d %f %f \n", msize, gflops / time_best, max_diff);
    }
    fclose(fptr); 
}