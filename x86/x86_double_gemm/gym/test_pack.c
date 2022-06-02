#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 

#define mat(i, j) mat[(i)*(ldm) + (j)]
#define A(i, j) A[(i)*(lda) + (j)]
#define B(i, j) B[(i)*(ldb) + (j)]

void random_matrix(int m, int n, double *mat, int ldm){
    double drand48();

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            // mat(i, j) = 2.0 * drand48( ) - 1.0;
            mat(i, j) = (double) (i + 1);
    }
    // print_matrix(m, n, mat, n);
}

void print_matrix(int m, int n, double *mat, int ldm){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            //std::cout<<mat(i, j)<<" ";
            printf("%f\t", mat(i, j));
        }
        // std::cout<<std::endl;
        printf("\n");
    }
    printf("\n");   
}


int test_packA() {
    const int m = 8, k = 4;
    int lda = k;
    double *A = (double*) malloc(m * k * sizeof(double));
    random_matrix(m, k, A, lda);
    print_matrix(m, k, A, lda);

    double *packedA = (double*) malloc(m * k * sizeof(double));
    double *packedA_ptr = packedA;
    for (int i = 0; i < m; i += 4) {
        for (int p = 0; p < k; ++p) {
            *packedA_ptr = A(i, p);
            *(packedA_ptr + 1) = A(i + 1, p);
            *(packedA_ptr + 2) = A(i + 2, p);
            *(packedA_ptr + 3) = A(i + 3, p);
            packedA_ptr += 4;
        }
    }

    for (int i = 0; i < m * k; ++i) {
        printf("%f\n", packedA[i]);
    }
    
    return 0;
}

int test_packB() {
    const int k = 4, n = 8;
    int ldb = n;
    double *B = (double*) malloc(k * n * sizeof(double));
    random_matrix(k, n, B, ldb);
    print_matrix(k, n, B, ldb);

    double *packedB = (double*) malloc(k * n * sizeof(double));
    double *packedB_ptr = packedB;
    for (int j = 0; j < n; j += 4) {
        for (int p = 0; p < k; ++p) {
            *packedB_ptr = B(p, j);
            *(packedB_ptr + 1) = B(p, j + 1);
            *(packedB_ptr + 2) = B(p, j + 2);
            *(packedB_ptr + 3) = B(p, j + 3);

            packedB_ptr += 4;
        }
    }

    for (int i = 0; i < k * n; ++i) {
        printf("%f\n", packedB[i]);
    }
    
    return 0;
}

int main() {
    // test_packA();
    test_packB();

    return 0;
}




// double *packedA = (double*) malloc(m * k * sizeof(double));
