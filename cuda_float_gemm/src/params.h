
// #define m 

// About leading demension: https://www.youtube.com/watch?v=PhjildK5oO8
// in row-major matrix, ld is the width of the matrix

// Note that I re-implement in a [row-major] manner
#define A(i, j) A[(i)*(lda) + (j)]
#define B(i, j) B[(i)*(ldb) + (j)]
#define C(i, j) C[(i)*(ldc) + (j)]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]

#define d_A(x, y) d_A[(y)*(lda) + x]
#define d_B(x, y) d_B[(y)*(ldb) + x]
#define d_C(x, y) d_C[(y)*(ldc) + x]

// #define A_sub(x, y) A_sub[(y)*(lda) + x]
// #define B_sub(x, y) B_sub[(y)*(ldb) + x]
// #define C_sub(x, y) C_sub[(y)*(ldc) + x]

// Note that the brackets in A[_(i)_*lda + j] REALLY matters!
// otherwise when you call A[p+1, j], you will get A[p+1*lda, j], rather than A[(p+1)*lda, j]!!!

#define ASIZE(type) (sizeof(type) * m * k)
#define BSIZE(type) (sizeof(type) * k * n)
#define CSIZE(type) (sizeof(type) * m * n)

// https://stackoverflow.com/a/3437433/10096987
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// #define mBlockSize 40
// #define kBlockSize 40
// #define nBlockSize 40
// #define mBlockSize 256
// #define kBlockSize 128
// #define nBlockSize 128

// #define BLOCK_SIZE 16