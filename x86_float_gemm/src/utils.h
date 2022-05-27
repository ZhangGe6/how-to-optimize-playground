#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 

void random_matrix(float *mat, int M, int N);
void zero_matrix(float *mat, int M, int N);
void print_matrix(float *mat, int M, int N);
void print_local_matrix(float *mat, int M, int N, int start_row, int start_col, int row_cover, int col_cover);
float compare_matrix(float *mat, float *mat2, int M, int N);