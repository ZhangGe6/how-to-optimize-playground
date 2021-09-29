// params.h
// Note that I re-implement in a [row-major] manner
#define A(i, j) A[i*lda + j]
#define B(i, j) B[i*ldb + j]
#define C(i, j) C[i*ldc + j]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]