
// #define m 


// // define leading dimension
// #define lda 1000
// #define ldb 1000
// #define ldc 1000
// #define ldm 1000

// create macros so that the matrices are stored in [row-major] order
#define A(i, j) A[i*lda + j]
#define B(i, j) B[i*ldb + j]
#define C(i, j) C[i*ldc + j]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]

