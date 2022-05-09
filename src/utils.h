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


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
// __m128d is a data type that the compiler will hopefully store in a XMM 128 bit register when optimizing
// https://stackoverflow.com/questions/53757633/what-is-m128d
// https://zhuanlan.zhihu.com/p/55327037
// 2 double, 4 float, ...
typedef union {
    __m128d reg;    
    double value[2];
} v2d_regv;