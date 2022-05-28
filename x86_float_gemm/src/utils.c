#include "params.h"
#include "utils.h"

// https://stackoverflow.com/a/13409133/10096987
void random_matrix(float *mat, int M, int N, int ldm){
    float a = 10000.0;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j)
            // mat(i, j) = (float)rand()/(float)(RAND_MAX/a);
            mat(i, j) = (float) (i);
    }
    // print_matrix(M, N, mat, N);
}

void zero_matrix(float *mat, int M, int N, int ldm){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j)
            mat(i, j) = (float) 0;
    }
    // print_matrix(M, N, mat, N);
}

void print_matrix(float *mat, int M, int N, int ldm){
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

void print_local_matrix(float *mat, int M, int N, int start_row, int start_col, int row_cover, int col_cover, int ldm){
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

float compare_matrix(float *mat, float *mat2, int M, int N, int ldm){
    float max_diff = 0, diff;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            diff = fabs(mat(i, j) - mat2(i, j));
            max_diff = diff > max_diff ? diff : max_diff;
            // if (diff > 0){
            //     printf("unequal at [%d, %d], values are %f and %f, diference is %f\n", i, j, mat(i, j), mat2(i, j), diff);
            //     // exit(1);
            // }
        }
    }

    return max_diff;
}

float max_value(float *mat, int M, int N, int ldm){
    float max_value = -DBL_MAX;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            if (mat(i, j) > max_value)  max_value = mat(i, j);
        }
    }

    return max_value;
}

// float compare_matrix(float *mat, float *mat2, int M, int N, int ldm){
//     float max_diff = 0, diff;
//     int max_y, max_x;
//     int i, j;
//     for (i = 0; i < M; ++i){
//         for (j = 0; j < N; ++j){
//             // diff = abs(mat(i, j) - mat2(i, j));
//             diff = fabs(mat(i, j) - mat2(i, j));
//             if (i == 0 && j == 0) {
//                 printf("[%d, %d], mat %f, mat2 %f -> %f\n", i, j, mat(i, j), mat2(i, j), diff);
//             }
//             if (diff > max_diff) {
//                 max_diff = diff;
//                 max_y = i;
//                 max_x = j;
//             }
//         }
//     }
//     printf("max diff at [%d, %d], -> %f\n", max_y, max_x, max_diff);

//     return max_diff;
// }
