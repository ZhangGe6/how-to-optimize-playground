#include "params.h"
#include "utils.h"

// https://stackoverflow.com/a/13409133/10096987
void random_matrix(float *mat, int M, int N){
    float a = 5.0;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j)
            // mat(i, j) = (float)rand()/(float)(RAND_MAX/a);
            mat(i, j) = (float) (i + 1);
    }
    // print_matrix(M, N, mat, N);
}

void zero_matrix(float *mat, int M, int N){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j)
            mat(i, j) = (float) 0;
    }
    // print_matrix(M, N, mat, N);
}

void print_matrix(float *mat, int M, int N){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            //std::cout<<mat(i, j)<<" ";
            printf("%f\t", mat(i, j));
        }
        // std::cout<<std::endl;
        printf("\n");
    }
    printf("\n");   
}

void print_local_matrix(float *mat, int M, int N, int start_row, int start_col, int row_cover, int col_cover){
    for (int i = start_row; i < start_row + row_cover; ++i){
        for (int j = start_col; j < start_col + col_cover; ++j){
            //std::cout<<mat(i, j)<<" ";
            printf("%f\t", mat(i, j));
        }
        // std::cout<<std::endl;
        printf("\n");
    }
    printf("\n");   
}

float compare_matrix(float *mat, float *mat2, int M, int N){
    float max_diff = 0, diff;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            diff = abs(mat(i, j) - mat2(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
        }
    }

    return max_diff;
}
