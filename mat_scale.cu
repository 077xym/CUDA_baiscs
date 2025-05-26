#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 6000
#define N 8000
#define BLOCK_SIZE 32

// cpu matrix scaling
void mat_scale_cpu(float *a, int row, int col, float scale) {
    for (int i = 0; i < row*col; i++) {
        a[i] *= scale;
    }
}

/*
    gpu matrix scaling
    we don't need to flatten our thread into list in this case, we can flatten
    our thread to global i, j.

    for rows, blockIdx.y * blockDim.y + threadIdx.y
    for cols, blockIdx.x * blockDim.x + threadIdx.x

    for thread[row, col], it is responsible for a[row][col], to flatten, it is the row * cols + col th element of a
*/ 
__global__ void mat_scale_gpu(float *a, int rows, int cols, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row * col < rows * cols) {
        a[row * cols + col] *= scale;
    }
}

// randomly init a matrix
void init_mat(float *a, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

// measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // generate variables for host and device
    float *h_a, *h_a_gpu;
    float *d_a;
    size_t size = M * N * sizeof(float);

    // allocate memory
    h_a = (float*)malloc(size);
    h_a_gpu = (float*)malloc(size);
    cudaMalloc(&d_a, size);

    // generate random matrix and copy to device
    srand(time(NULL));
    init_mat(h_a, M, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    float scale = 0.5;

    // cpu benchmark
    float cpu_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        float stime_cpu = get_time();
        mat_scale_cpu(h_a, M, N, scale);
        float etime_cpu = get_time();
        cpu_time += etime_cpu - stime_cpu;
    }
    printf("cpu benchmark: %f milliseconds\n", cpu_time / 20.0 * 1000.0);

    // gpu benchmark
    float gpu_time = 0.0f;
    // remember the CUDA convention, we need the first element to be col
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    for (int i = 0; i < 20; i++) {
        float stime_gpu = get_time();
        mat_scale_gpu<<<grid_size, block_size>>>(d_a, M, N, scale);
        cudaDeviceSynchronize();
        float etime_gpu = get_time();
        gpu_time += etime_gpu - stime_gpu;
    }
    printf("gpu benchmark: %f milliseconds\n", gpu_time / 20.0 * 1000.0);

    // correctness check
    bool correct = true;
    cudaMemcpy(h_a_gpu, d_a, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_a[i] - h_a_gpu[i]) > 1e-3) {
            correct = false;
            printf("(%d, %d) is not equal, h_a: %f, h_a_gpu: %f\n", i / N, i % N, h_a[i], h_a_gpu[i]);
            break;
        }
    }
    printf("correctness check result: %s\n", correct ? "correct" : "incorrect");

    // free mem
    free(h_a);
    free(h_a_gpu);
    cudaFree(d_a);
}
