
// #define m 


// // define leading dimension
// #define lda 1000
// #define ldb 1000
// #define ldc 1000
// #define ldm 1000

// Note that I re-implement in a [row-major] manner
#define A(i, j) A[(i)*lda + j]
#define B(i, j) B[(i)*ldb + j]
#define C(i, j) C[(i)*ldc + j]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]

// Note that the brackets in A[_(i)_*lda + j] REALLY matters!
// otherwise when you call A[p+1, j], you will get A[p+1*lda, j], rather than A[(p+1)*lda, j]!!!