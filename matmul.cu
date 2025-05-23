#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // Number of rows in A and C
#define K 512 // Number of cols in A and rows in B
#define N 256 // Number of cols in B and cols in C
#define BLOCK_SIZE 32 // block dim, which is 32 by 32

// CPU matrix multiplication
void matmul_cpu(float *a, float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/*
    GPU matrix multiplication
    Where each thread (i, j) is responsible for calculating row i of a times col j of b

    So what we need is to calculate the global i and j of threads using blockIdx and threadIdx

    Note!!! for both grid and block, the convention for CUDA is:
    suppose grid is in dim (a, b, c), then it is actually c by b by a. i.e, when we set
    the grid dim, the first axes denotes the fastest changing axes while the last axes denotes
    the depth.
    Meaning that
    blockIdx.x is apartment number
    blockIdx.y is floor
    blockIdx.z is building

    If you are to visualize your grid matrix, if blockIdx.x = 2, blockIdx.y = 3, then
    it is actually the (3, 2) block in your grid matrix

    Back to this problem, let's first compute global i, which is the row
    So we should look into blockIdx.y. It is pretty hard to explain how I
    come up with this formula, it is always recommended to visualize it.
    i = blockIdx.y * blockDim.y + threadIdx.y

    for j, we apply similar idea
    j = blockIdx.x * blockDim.x + threadIdx.x
*/ 

__global__ void matmul_gpu(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int l = 0; l < k; l++) {
        sum += a[row * k + l] * b[l * n + col];
    }
    c[row * n + col] = sum;
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// main process
int main() {
    // create variables holding data for host and devices
    float *h_a, *h_b, *h_c, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // allocate memory on cpu
    h_a = (float *)malloc(size_A);
    h_b = (float *)malloc(size_B);
    h_c = (float *)malloc(size_C);
    h_c_gpu = (float *)malloc(size_C);

    // initialize matrix
    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // allocate device mem (note the input is pointer of pointer)
    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    // Send h_a, h_b, h_c_gpu to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    // CPU benchmark
    double cpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        matmul_cpu(h_a, h_b, h_c, M, K, N);
        double etime = get_time();
        cpu_time = etime - stime;
    }
    printf("CPU execution time: %f seconds\n", cpu_time / 20.0);

    // set up grid and Block
    /*
        Note, the convention is to set the grid in reverse
        that is (col, row, depth)
    */
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

    // GPU benchmark
    double gpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        matmul_gpu<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        double etime = get_time();
        gpu_time += etime - stime;
    }
    printf("GPU execution time: %f seconds\n", gpu_time / 20.0);

    // check correctness
    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_c[i] - h_c_gpu[i]) > 1e-3) {
            correct = false;
            printf("(%d, %d) not equal. h_c: %f, h_c_gpu: %f\n", i / M, i % M, h_c[i], h_c_gpu[i]);
            break;
        }
    }
    printf("result is %s\n", correct ? "correct" : "incorrect");

    // free up memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
