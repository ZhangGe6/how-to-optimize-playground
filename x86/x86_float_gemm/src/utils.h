#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 

void random_matrix(float *mat, int M, int N, int ldm);
void zero_matrix(float *mat, int M, int N, int ldm);
void print_matrix(float *mat, int M, int N, int ldm);
void print_local_matrix(float *mat, int M, int N, int start_row, int start_col, int row_cover, int col_cover, int ldm);
float compare_matrix(float *mat, float *mat2, int M, int N, int ldm);
float max_value(float *mat, int M, int N, int ldm);