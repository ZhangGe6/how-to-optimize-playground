#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "assert.h" 

void random_matrix( int m, int n, double *mat, int ldm);
void zero_matrix( int m, int n, double *mat, int ldm);
void print_matrix(int m, int n, double *mat, int ldm);
double compare_matrix(int m, int n, double *mat1, double *mat2, int ldm);