#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "assert.h" 

void allocate_space(int m, int k, int n, double *matA, double *matB, double *matC);
void random_matrix( int m, int n, double *mat, int ldm);
void zero_matrix( int m, int n, double *mat, int ldm);
void print_matrix(int m, int n, double *mat, int ldm);
double compare_matrix(int m, int n, double *mat1, double *mat2, int ldm);
double dclock();

// https://stackoverflow.com/a/3437433/10096987
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))