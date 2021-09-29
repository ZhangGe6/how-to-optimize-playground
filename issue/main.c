// main.c
#include "params.h"
#include "utils.h"
#include "MMult.h"

int main(){
    //for (int msize = 40; msize <= 800; msize += 40){
    for (int msize = 4; msize <= 4; msize += 4){     // small matrix for debugging
        double *A, *B, *C_base, *C_optim;
        int m = msize, k = msize, n = msize;
        int lda = k, ldb = n, ldc = n;

        A = (double*) malloc(m * k * sizeof(double));
        B = (double*) malloc(k * n * sizeof(double));
        C_base = (double*) malloc(m * n * sizeof(double));
        C_optim = (double*) malloc(m * n * sizeof(double));
        random_matrix(m, k, A, lda);
        random_matrix(k, n, B, ldb);
        zero_matrix(m, n, C_base, ldc);
        zero_matrix(m, n, C_optim, ldc);

        printf("A\n");
        print_matrix(m, k, A, lda);
        printf("B\n");
        print_matrix(k, n, B, ldb);

        MMult_base(m, k, n, A, B, C_base, lda, ldb, ldc);          // store the baseline result into C_base
        // MMult_unroll_inner(m, k, n, A, B, C_optim, lda, ldb, ldc); 
        MMult_unroll_inner(m, k, n, A, B, C_optim, lda, ldb, ldc); // store the optimized result into C_optim

        printf("C_base\n");
        print_matrix(m, n, C_base, ldc);
        printf("C_optim\n");
        print_matrix(m, n, C_optim, ldc);

        double max_diff = compare_matrix(m, n, C_base, C_optim, ldc); // compare C_base and C_optim
        assert(max_diff == 0);                                     
    }
}