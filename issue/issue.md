# Inner unloop leads to wrong results

Hi, Thanks for this excellent work! I tried the unroll trick as [Optimization2](https://github.com/flame/how-to-optimize-gemm/wiki/Optimization2) suggests, which unrolls the outer loop, i.e., j. This works and gives me the right caculation result (although no speed boost on my machine, too). However, when I tried further to unroll the inner loop, i.e., i or p, an caculation error occurs. 

This is my code:

```cpp
// params.h
// Note that I re-implement in a [row-major] manner
#define A(i, j) A[i*lda + j]
#define B(i, j) B[i*ldb + j]
#define C(i, j) C[i*ldc + j]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]
```

```cpp
// MMult_base.c
#include "params.h"
#include "MMult.h"

void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int i = 0; i < m; ++i){
    for (int j  = 0; j < n; ++j){
      for (int p = 0; p < k; ++p)
        C(i, j) += A(i, p) * B(p, j);
    }
  }
}
```

```cpp
// MMult_unroll.c
#include "params.h"
#include "MMult.h"
#include <stdio.h>

// This gives a correct caculation result
void MMult_unroll(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 2){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; ++p){
        C(i, j) += A(i, p) * B(p, j);
        C(i, j + 1) += A(i, p) * B(p, j + 1);
      }
    }
  }
}

// However, when I tried inner loop, i or p, wrong result occurs
void MMult_unroll_inner(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc)
{
  for (int j = 0; j < n; j += 1){
    for (int i  = 0; i < m; i += 1){
      for (int p = 0; p < k; p += 2){

        C(i, j) += A(i, p) * B(p, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p, A(i, p), p, j, B(p, j));
        C(i, j) += A(i, p + 1) * B(p + 1, j);
        printf("C(%d, %d) added by [A(%d, %d)=%f] * [B(%d, %d)=%f]\n", i, j, i, p+1, A(i, p+1), p+1, j, B(p+1, j));
        
      }
      printf("\n");
    }
  }
}
```

```cpp
// main.c
#include "params.h"
#include "utils.h"
#include "MMult.h"

int main(){
    //for (int msize = 40; msize <= 800; msize += 40){
    for (int msize = 4; msize <= 4; msize += 4){     // small matrix for debugging
        double *A, *B, *C_base, *C_optim;
        int m = msize, k = msize, n = msize;
        int lda = k, ldb = n, ldc = n;

        A = (double*) malloc(m * k * sizeof(double));
        B = (double*) malloc(k * n * sizeof(double));
        C_base = (double*) malloc(m * n * sizeof(double));
        C_optim = (double*) malloc(m * n * sizeof(double));
        random_matrix(m, k, A, lda);
        random_matrix(k, n, B, ldb);
        zero_matrix(m, n, C_base, ldc);
        zero_matrix(m, n, C_optim, ldc);

        printf("A\n");
        print_matrix(m, k, A, lda);
        printf("B\n");
        print_matrix(k, n, B, ldb);

        MMult_base(m, k, n, A, B, C_base, lda, ldb, ldc);          // store the baseline result into C_base
        // MMult_unroll_inner(m, k, n, A, B, C_optim, lda, ldb, ldc); 
        MMult_unroll_inner(m, k, n, A, B, C_optim, lda, ldb, ldc); // store the optimized result into C_optim

        printf("C_base\n");
        print_matrix(m, n, C_base, ldc);
        printf("C_optim\n");
        print_matrix(m, n, C_optim, ldc);

        double max_diff = compare_matrix(m, n, C_base, C_optim, ldc); // compare C_base and C_optim
        assert(max_diff == 0);                                     
    }
}
```
header files,  util functions, and makefile I use to compile are here for helping debugging
```cpp
// MMult.h
void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
void MMult_unroll(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
void MMult_unroll_inner(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);

```

```cpp
// utils.h
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "assert.h" 

void random_matrix( int m, int n, double *mat, int ldm);
void zero_matrix( int m, int n, double *mat, int ldm);
void print_matrix(int m, int n, double *mat, int ldm);
double compare_matrix(int m, int n, double *mat1, double *mat2, int ldm);

```

```cpp
// utils.c
#include "params.h"
#include "utils.h"

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
```

```makefile
# makefile

CC         := gcc
CFLAGS     := -O2 -Wall -msse3
LDFLAGS    := -lm


UTIL := utils.o
MulMethods := $(patsubst %.c, %.o, $(wildcard MMult*.c))
TEST_OBJS  := main.o $(MulMethods)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all:
	make clean
	echo $(MulMethods)
	make test_MMult.x
	make clean_tmp

test_MMult.x: $(TEST_OBJS) $(UTIL) params.h
	$(CC) $(TEST_OBJS) $(UTIL) $(LDFLAGS) $(BLAS_LIB) -o $(TEST_BIN) $@ 

clean:
	rm -f *.o *.x
clean_tmp:
	rm -f *.o

```
This has confused me a lot. Can anyone help me? 

Thanks in advance. 