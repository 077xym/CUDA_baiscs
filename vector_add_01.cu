#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cmath>

// define the vector size
#define N 10000000
// define the block size (how many threads within a block)
#define BLOCK_SIZE 256

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for GPU vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    /*
        grid dim: ceil(N/BLOCK_SIZE) x 1 x 1
        block dim: BLOCK_SIZE x 1 x 1
        
        workflow:
        obtain the offset of each thread, which is the idx to access a and b
        remember to check whether the offset is larger than n or not

        Since whatever the grid and dim size we chose, they will implicitly be transformed to
        3D, we can always use what is shown in 01_index.cu to get offset. However, we reduce
        our operation counts because we know some of the dim will be statically 0.
    */
    int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_offset < n) {
        c[thread_offset] = a[thread_offset] + b[thread_offset];
    }   
}

// CUDA kernel for GPU vector addition 3D
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int n) {
    int block_offset = 
        blockIdx.x * gridDim.y * gridDim.z +
        blockIdx.y * gridDim.z +
        blockIdx.z;

    int thread_offset = 
        block_offset * blockDim.x +
        threadIdx.x;
    
    if (thread_offset < n) {
        c[thread_offset] = a[thread_offset] + b[thread_offset];
    }
}

// Init vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// a complete view of how to implement vector addition using CUDA
int main() {
    
    // claim the host data variables, remember, these are the data stored in cpu
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu, *h_c_gpu_3d;
    // claim the device data variables, these are the data stored in gpu, copied from cpu
    float *d_a, *d_b, *d_c, *d_c_3d;
    // size_t, how many mem are going to be allocated
    size_t size = N * sizeof(float);

    // Allocate host memory (malloc returns void*, so we need to cast its type)
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);
    h_c_gpu_3d = (float *)malloc(size);

    // randomly initialize the vector
    srand(time(NULL)); // set the seed
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_c_3d, size);

    // copy data from cpu to gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /*
        define grid dimension. In this code snippet, we want our grid and block both have 
        1D dimensions. Note, instead of being 1 x BLOCK_SIZE, the actual block dim is implicitly
        BLOCK_SIZE x 1 x 1. So as the grid dim. The number in between <<< >>> tells the nvcc
        the grid and block dim.
    */
    /*
        initially, I wrote int grid_size = ceil(N / BLOCK_SIZE)
        This is wrong, as N and BLOCK_SIZE are both int, N / BLOCK_SIZE has already been floored
        making ceil useless here
    */
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("%d\n", grid_size);
    printf("Benchmarking GPU implementation...\n");
    double gpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        vector_add_gpu<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double etime = get_time();
        gpu_time += etime - stime;
    }
    printf("GPU average time: %f seconds\n", gpu_time / 20.0);

    /*
        Let's define grid to be 2D (cause (N + BLOCK_SIZE - 1) / BLOCK_SIZE can only be divisible by 3)
        but we can have a 1 at any axes to make it a 3D
    */
    dim3 grid_dim(3, 13021, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    printf("Benchmarking GPU-3D implementation...\n");
    double gpu_time_3d = 0.0;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        vector_add_gpu_3d<<<grid_dim, block_dim>>>(d_a, d_b, d_c_3d, N);
        cudaDeviceSynchronize();
        double etime = get_time();
        gpu_time_3d += etime - stime;
    }
    printf("GPU average time: %f seconds\n", gpu_time_3d / 20.0);


    // execute cpu-version vector addition, and report the time
    printf("Benchmarking CPU implementation...\n");
    double cpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double etime = get_time();
        cpu_time += etime - stime;
    }
    printf("CPU average time: %f seconds\n", cpu_time / 20.0);

    // check results are equal
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-2) {
            correct = false;
            printf("index: %d incorrect. cpu: %f, gpu: %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }
    printf("Results 1d are %s\n", correct ? "correct" : "incorrect");

    // check results are equal
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-2) {
            correct_3d = false;
            printf("index: %d incorrect. cpu: %f, gpu: %f\n", i, h_c_cpu[i], h_c_gpu_3d[i]);
            break;
        }
    }
    printf("Results 3d are %s\n", correct_3d ? "correct" : "incorrect");
    printf("%f", h_c_gpu_3d[769]);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_3d);

    return 0;
}
