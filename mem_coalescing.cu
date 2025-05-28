#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 4000
#define N 4000
#define BLOCK_SIZE 256

/*
    In this snippet, I plan to test the effect of memory coalescing
    For a matrix of size 8000 by 8000, we have two different set of threads where
    1. set 1: thread responsible for scaling row wisely
    2. set 2: thread responsible for scaling col wisely (coalesced)
*/

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void mat_scale_non_col(float *a, int m, int n, float scale) {
    /*
        In this case, we have each thread responsible for ith row
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] *= scale;
        }
    }
}


__global__ void mat_scale_col(float *a, int m, int n, float scale) {
    /*
        In this case, we have each thread responsible for j^th row
    */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        for (int i = 0; i < m; i++) {
            a[i * n + j] *= scale;
        }
    }
}

int main() {
    float *h_a, *h_a_col, *h_a_non_col;
    float *d_a_col, *d_a_non_col;
    size_t size = M * N * sizeof(float);
    float scale = 0.8;

    // allocate host data
    h_a = (float *)malloc(size);
    h_a_col = (float *)malloc(size);
    h_a_non_col = (float *)malloc(size);

    // allocate device data
    cudaMalloc(&d_a_col, size);
    cudaMalloc(&d_a_non_col, size);

    // init mat and assign to device
    init_mat(h_a, M, N);
    cudaMemcpy(d_a_col, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_non_col, h_a, size, cudaMemcpyHostToDevice);

    // benchmark for non-coalsced
    int grid_size_non_col = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double gpu_non_col_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double gpu_non_col_stime = get_time();
        mat_scale_non_col<<<grid_size_non_col, BLOCK_SIZE>>>(d_a_non_col, M, N, scale);
        cudaDeviceSynchronize();
        double gpu_non_col_etime = get_time();
        gpu_non_col_time += gpu_non_col_etime - gpu_non_col_stime;
    }
    printf("Non-coalsced: %f microseconds\n", gpu_non_col_time / 20.0 * 1000000.0);

    // benchmark for coalsced
    int grid_size_col = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double gpu_col_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double gpu_col_stime = get_time();
        mat_scale_col<<<grid_size_col, BLOCK_SIZE>>>(d_a_col, M, N, scale);
        cudaDeviceSynchronize();
        double gpu_col_etime = get_time();
        gpu_col_time = gpu_col_etime - gpu_col_stime;
    }
    printf("Coalsced: %f microseconds\n", gpu_col_time / 20.0 * 1000000.0);

    // check correctness
    cudaMemcpy(h_a_col, d_a_col, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_non_col, d_a_non_col, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (h_a_col[i] != h_a_non_col[i]) {
            printf("Values unequal. (%d, %d), col: %f, non-col: %f\n", i / M, i % N, h_a_col[i], h_a_non_col[i]);
            correct = false;
            break;
        }
    }
    printf("correctness check: %s", correct ? "correct\n" : "incorrect");

    // free memory
    free(h_a);
    free(h_a_col);
    free(h_a_non_col);
    cudaFree(d_a_col);
    cudaFree(d_a_non_col);
}
