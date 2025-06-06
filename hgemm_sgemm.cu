#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define M 14828
#define K 8928
#define N 10697
#define TILE_SIZE 32

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

void print_mat(float *mat, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f", mat[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float) rand() / RAND_MAX;    
    }
}

__global__ void matmul_tiling(float* a, float* b, float* c, int m, int k, int n) {
    __shared__ float TILE_A[TILE_SIZE][TILE_SIZE]; 
    __shared__ float TILE_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        /*
            each thread within a block is responsible for loading one element from the corresponding tile a and b
        */
        if (row < m && tile * TILE_SIZE + tx < k) {
            TILE_A[ty][tx] = a[row * k + tile * TILE_SIZE + tx];
        } else {
            TILE_A[ty][tx] = 0.0f;
        }

        if (tile * TILE_SIZE + ty < k && col < n) {
            TILE_B[ty][tx] = b[(tile * TILE_SIZE + ty) * n + col];
        } else {
            TILE_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int l = 0; l < TILE_SIZE; ++l) {
            sum += TILE_A[ty][l] * TILE_B[l][tx];
        }

        __syncthreads();    
    }

    if (row < m & col < n) {
        c[row * n + col] = sum;
    }
}

int main() {
    float *h_a, *h_b, *h_tile, *h_cublas;
    float *d_a, *d_b, *d_c_tile, *d_c_cublas;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_a = (float *)malloc(size_A);
    h_b = (float *)malloc(size_B);
    h_tile = (float *)malloc(size_C);
    h_cublas = (float *)malloc(size_C);
    
    srand(time(NULL));
    init_mat(h_a, M, K);
    init_mat(h_b, K, N);

    // CUDA setup

    CHECK_CUDA(cudaMalloc(&d_a, size_A));
    CHECK_CUDA(cudaMalloc(&d_b, size_B));
    CHECK_CUDA(cudaMalloc(&d_c_tile, size_C));
    CHECK_CUDA(cudaMalloc(&d_c_cublas, size_C));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice));

    // benchmarking for tiling
    dim3 gird_dim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    double tile_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double stime = get_time();
        matmul_tiling<<<gird_dim, block_dim>>>(d_a, d_b, d_c_tile, M, K, N);
        cudaDeviceSynchronize();
        double etime = get_time();
        tile_time += etime - stime;
    }
    printf("tiling: %f ms\n", tile_time / 20.0 * 1000.0);


    // CUBLAS setup, we need a handle, which can be considered as a session
    // to record this operation's state
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // see my document on how to handle row-major and col-major issues
    float alpha = 1.0f, beta = 0.0f;
    double cublas_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double cublas_stime = get_time();
        CHECK_CUBLAS(cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            N, M, K, 
            &alpha,
            d_b, N,
            d_a, K, 
            &beta,
            d_c_cublas, N));
        cudaDeviceSynchronize();
        double cublas_etime = get_time();
        cublas_time += cublas_etime - cublas_stime;
    }
    printf("cublas sgemm %f ms\n", cublas_time / 20.0 * 1000.0);
    
    CHECK_CUDA(cudaMemcpy(h_cublas, d_c_cublas, size_C, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(h_tile, d_c_tile, size_C, cudaMemcpyDeviceToHost));

    free(h_a);
    free(h_b);
    free(h_cublas);
    free(h_tile);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_tile));
    CHECK_CUDA(cudaFree(d_c_cublas));
}
