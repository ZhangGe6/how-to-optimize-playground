#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)

// About leading demension: https://www.youtube.com/watch?v=PhjildK5oO8
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// https://stackoverflow.com/a/3437433/10096987
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])