
// #include <iostream>
#include "params.h"
#include "utils.h"

void allocate_space(int m, int k, int n, float *matA, float *matB, float *matC){
    matA = (float*) malloc(m * k * sizeof(float));
    matB = (float*) malloc(k * n * sizeof(float));
    matC = (float*) malloc(m * n * sizeof(float));
    // print_matrix(m, n, mat, n);
}

void random_matrix(int m, int n, float *mat, int ldm){
    float drand48();

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            // mat(i, j) = 2.0 * drand48( ) - 1.0;
            // mat(i, j) = (float) (i + 1);
            mat(i, j) = 1;
    }
    // print_matrix(m, n, mat, n);
}

void zero_matrix(int m, int n, float *mat, int ldm){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat(i, j) = (float) 0;
    }
    // print_matrix(m, n, mat, n);
}

void print_matrix(int m, int n, float *mat, int ldm){
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

float compare_matrix(int m, int n, float *mat, float *mat2, int ldm){
    float max_diff = 0, diff;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            diff = abs(mat(i, j) - mat2(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
        }
    }

    return max_diff;
}

/* Adapted from the bl2_clock() routine in the BLIS library */
static float gtod_ref_time_sec = 0.0;
float dclock()
{
        float the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( float ) tv.tv_sec;

        norm_sec = ( float ) tv.tv_sec - gtod_ref_time_sec;
        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}
