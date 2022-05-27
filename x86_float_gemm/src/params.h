// About leading demension: https://www.youtube.com/watch?v=PhjildK5oO8
// Note that I implement in a [row-major] manner
#define A(i, j) A[(i)*(K) + (j)]
#define B(i, j) B[(i)*(N) + (j)]
#define C(i, j) C[(i)*(N) + (j)]
#define mat(i, j) mat[(i)*(N) + (j)]
#define mat2(i, j) mat2[(i)*(N) + (j)]

// // Note that the brackets in A[_(i)_*lda + j] REALLY matters!
// // otherwise when you call A[p+1, j], you will get A[p+1*lda, j], rather than A[(p+1)*lda, j]!!!

// https://stackoverflow.com/a/3437433/10096987
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
// __m128d is a data type that the compiler will hopefully store in a XMM 128 bit register when optimizing
// https://stackoverflow.com/questions/53757633/what-is-m128d
// https://zhuanlan.zhihu.com/p/55327037
// 2 double, 4 float, ...
// union: https://www.runoob.com/cprogramming/c-unions.html
typedef union {
    __m128d reg;    
    double value[2];
} v2d_regv;

// #define blockSize 40
// #define mBlockSize 40
// #define kBlockSize 40
// #define nBlockSize 40
// // #define mBlockSize 256
// // #define kBlockSize 128
// #define nBlockSize 128