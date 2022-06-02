#include "utils.h"

void random_matrix(float *mat, int m, int n){
    // https://techiedelight.com/compiler/
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat[OFFSET(i, j, n)] = dist(gen);
            // mat[OFFSET(i, j, n)] = 1;
    }
    // print_matrix(m, n, mat, n);
}

void zero_matrix(float *mat, int m, int n){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat[OFFSET(i, j, n)] = (float) 0;
    }
    // print_matrix(m, n, mat, n);
}

void print_matrix(float *mat, int m, int n){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            //std::cout<<mat(i, j)<<" ";
            printf("%f\t", mat[OFFSET(i, j, n)]);
        }
        // std::cout<<std::endl;
        printf("\n");
    }
    printf("\n");   
}

float compare_matrix(float *mat1, float *mat2, int m, int n){
    float diff, max_diff = 0;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            diff = fabs(mat1[OFFSET(i, j, n)] - mat2[OFFSET(i, j, n)]);
            max_diff = diff > max_diff ? diff : max_diff;
        }
    }

    return max_diff;
}

// https://github1s.com/tpoisonooo/how-to-optimize-gemm/blob/HEAD/cuda/test_MMult.cpp
void print_gpu_info() {
    // print gpu info
    cudaDeviceProp deviceProp;
    int devID = 0;
    checkCudaErrors(cudaSetDevice(devID));
    auto error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
            __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("\nGPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
            deviceProp.name, deviceProp.major, deviceProp.minor);
}
