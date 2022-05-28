// About leading demension: https://www.youtube.com/watch?v=PhjildK5oO8
// Note that I implement in a [row-major] manner
#define A(i, j) A[(i)*(lda) + (j)]
#define B(i, j) B[(i)*(ldb) + (j)]
#define C(i, j) C[(i)*(ldc) + (j)]
#define mat(i, j) mat[(i)*(ldm) + (j)]
#define mat2(i, j) mat2[(i)*(ldm) + (j)]

// // Note that the brackets in A[_(i)_*lda + j] REALLY matters!
// // otherwise when you call A[p+1, j], you will get A[p+1*lda, j], rather than A[(p+1)*lda, j]!!!

// https://stackoverflow.com/a/3437433/10096987
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3

// https://zhuanlan.zhihu.com/p/55327037
// union: https://www.runoob.com/cprogramming/c-unions.html
typedef union {
    __m128 reg;    
    float value[4];
} v2f_regv;

// #define blockSize 40
// #define mBlockSize 40
// #define kBlockSize 40
// #define nBlockSize 40
// // #define mBlockSize 256
// // #define kBlockSize 128
// #define nBlockSize 128