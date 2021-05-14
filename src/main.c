// #include <iostream>
#include "params.h"
#include "utils.h"
#include "MMult.h"


int main(){
    double *A, *B, *C;
    int m = 2, k = 2, n = 2;
    int lda = k, ldb = n, ldc = n;

    /* each item of output require 2K floating point ops (multiply & add) and perform M*K times 
       See https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/ for more details*/
    double gflops = 2.0 * m * n * k * 1.0e-09;

    // allocate_space(m, k, &A); // 
    // allocate_space(k, n, &B);
    // allocate_space(m, k, n, &A, &B, &C);
    A = (double*) malloc(m * k * sizeof(double));
    B = (double*) malloc(k * n * sizeof(double));
    C = (double*) malloc(m * n * sizeof(double));
    random_matrix(m, k, A, lda);
    random_matrix(k, n, B, ldb);
    zero_matrix(m, n, C, ldc);
    
    // print_matrix(m, k, A, lda);
    // print_matrix(k, m, B, ldb);
    double time_s = dclock();
    Base_MMult(m, k, n, A, B, C, lda, ldb, ldc);
    // print_matrix(m, n, C, ldc);
    double elapse_time = dclock() - time_s;
    printf("%f", gflops / elapse_time);

    double max_diff = compare_matrix(m, n, C, C, ldc);
    // printf("%f", max_diff);






}