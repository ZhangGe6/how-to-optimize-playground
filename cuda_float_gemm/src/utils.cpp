
#include <iostream>
#include <random>
#include "params.h"
#include "utils.h"

void allocate_space(int m, int k, int n, float *matA, float *matB, float *matC){
    matA = (float*) malloc(m * k * sizeof(float));
    matB = (float*) malloc(k * n * sizeof(float));
    matC = (float*) malloc(m * n * sizeof(float));
    // print_matrix(m, n, mat, n);
}

void random_matrix(int m, int n, float *mat, int ldm){
    // https://techiedelight.com/compiler/
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(1, 10);

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat(i, j) = dist(gen);
            // mat(i, j) = (float) (i + 1);
            // mat(i, j) = 1;
    }
    // print_matrix(m, n, mat, n);
}

void zero_matrix(int m, int n, float *mat, int ldm){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j)
            mat(i, j) = (float) 0;
    }
    // print_matrix(m, n, mat, n);
}

void print_matrix(int m, int n, float *mat, int ldm){
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

float compare_matrix(int m, int n, float *mat, float *mat2, int ldm){
    float max_diff = 0, diff;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            diff = abs(mat(i, j) - mat2(i, j));
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
