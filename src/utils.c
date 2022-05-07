
// #include <iostream>
#include "params.h"
#include "utils.h"

void allocate_space(int m, int k, int n, double *matA, double *matB, double *matC){
    matA = (double*) malloc(m * k * sizeof(double));
    matB = (double*) malloc(k * n * sizeof(double));
    matC = (double*) malloc(m * n * sizeof(double));
    // print_matrix(m, n, mat, n);
}

void random_matrix(int m, int n, double *mat, int ldm){
    double drand48();

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            // mat(i, j) = 2.0 * drand48( ) - 1.0;
            mat(i, j) = (double) (i + 1);
    }
    // print_matrix(m, n, mat, n);
}

void zero_matrix(int m, int n, double *mat, int ldm){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat(i, j) = (double) 0;
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

double compare_matrix(int m, int n, double *mat, double *mat2, int ldm){
    double max_diff = 0, diff;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            diff = abs(mat(i, j) - mat2(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
        }
    }

    return max_diff;
}

/* Adapted from the bl2_clock() routine in the BLIS library */
static double gtod_ref_time_sec = 0.0;
double dclock()
{
        double the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( double ) tv.tv_sec;

        norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}

