#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
    We're going to use GPU for vector addition

    One good practice for vector addition on GPU is create a thread for elements at the same position
    on vector a and b

    Let's say we have both vector with length n
    -> number of threads: n
    -> threads per block: better to be a multiple of 32, say 256
    -> number of blocks: n / BLOCK_SIZE, a good trick to deal with ceiling without using type casting
    is (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    
    Now, we have
    dim3 block_dim = (1, 1, BLOCK_SIZE)
    dim3 gird_dim = (1, 1, (n + BLOCK_SIZE - 1) / BLOCK_SIZE)

    In this case, we don't actually need dim3 type as the dimension, BLOCK_SIZE and (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    is enough, as it will then be implicitly fulfill to (BLOCK_SIZE, 1, 1) and ((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1)

    Note that the 1 is added to the subsequent instead of previous. This is one nuance, as in CUDA convention,
    (x, y, z) actually means you have a dim z by y by x. better to pay attention when calculating global index of the threads
*/

#define N 500000000 // length of the vector
#define BLOCK_SIZE 256 // block_size

// cpu-way of computing vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// gpu-way of computing vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    /*
        How to get global threads? 
        blockIdx.x gives the block offset in this case, and threadIdx.x gives the offset within the block
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// randomly initialize vector
void init_vector(float *a, float n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)rand()/RAND_MAX;
    }
}

// measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // create variable for data on cpu and gpu
    float *h_a, *h_b, *h_c, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // allocate space to host data
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    // allocate space on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // initialize vector a and b, copy to device
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Benchmark for cpu calculation
    float cpu_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        float stime_cpu = get_time();
        vector_add_cpu(h_a, h_b, h_c, N);
        float etime_cpu = get_time();
        cpu_time += etime_cpu - stime_cpu;
        printf("cpu test-%d executed\n", i+1);
    }
    printf("cpu time: %f seconds\n", cpu_time / 20.0);

    // Benchmark for gpu calculation
    float gpu_time = 0.0f;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < 20; i++) {
        float stime_gpu = get_time();
        vector_add_gpu<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        float etime_gpu = get_time();
        gpu_time += etime_gpu - stime_gpu;
        printf("gpu test-%d executed\n", i+1);
    }
    printf("gpu time: %f seconds\n", gpu_time / 20.0);

    // check correctness
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_c_gpu[i]) {
            correct = false;
            printf("index: %d not equal. h_c: %f, h_c_gpu: %f\n", i, h_c[i], h_c_gpu[i]);
            break;
        }
    }
    printf("correctness check result: %s\n", correct ? "correct" : "incorrect");

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

