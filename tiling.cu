#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 16384
#define K 32768
#define N 32768
#define TILE_SIZE 32 // tile size, 32 * 32
#define BLOCK_SIZE 32 // block size, 32 * 32

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void matmul_naive(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m and col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += a[row * k + l] * b[l * n + col];
        }
        c[row * n + col] = sum;
    }
}

/*
    The threads_{x, y} in block_{i, j} are responsible for two things:
    1. tile-wisely load one scalar data from tile A and tile B
    2. calculate the dot product of xth row of A and y^th col of B

    The indexing is pretty complex here:
    
    We first need to notice that, block_{i, j} will requre tile_A_{i, 0-(K + TILE_SIZE - 1) / TILE_SIZE} and tile_B_{0-(K + TILE_SIZE - 1) / TILE_SIZE, j}
    We can then create a loop from tile = 0 to 0-(K + TILE_SIZE - 1) / TILE_SIZE
    At each tile, the thread first needs to load one data from corresponding tile_A_(i, tile) and tile_B_(tile, j).
    In this way, the loading process is intrinsically mem_coalesced as thread_{m, n} will load the local {m, n}^th element from both tiles
    Then after we have loaded all the data we need, we begin to compute the dot product
    In each tile, the thread is reponsible for computing the dot product of row m and col j locally in current tile_A and tile_B.

    sum = 0.0f
    i.e for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        load tile_A_{m, n} and tile_B_{m, n} into shared_memory
        wait all other threads in the block to finish loading

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[m][i] * tile_B[i][n];
        }
    }
    c[global index] = sum;

    To make things simple, let's define
    by = blockIdx.y, bx = blockIdx.x
    ty = threadIdx.y, tx = threadIdx.x

    Data uploading:
    For a certain thread_{ty, tx} in block_{by, bx}
    It is responsible for the elements of
    row = by * TILE_SIZE + ty 
    col = bx * TILE_SIZE + tx in C, which is row of A and col of B
    
    then we need to iterate through tile of A (by, tile) and B (tile, bx)
    
    At each tile, thread (ty, tx) will update one scalar data from tile A and tile B.
    The local index ty and tx is also the local index for shared memory, but the global index for the data of A and B is:

    A: row * K + tile * TILE_SIZE + tx
    B: (tile * TILE_SIZE + ty) * N + col

    Computing
    It is much simpler since in shared mem, the index aligned with the local index of the threads

    assign value
    We have global row and col

*/

__global__ void matmul_tile(float *a, float *b, float *c, int m, int k, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // simpler notation
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    // global index of row and col of current thread
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // if row and tile * TILE_SIZE + tx are within A, set A, else set nothing
        if (row < m && tile * TILE_SIZE + tx < k) {
            sharedA[ty][tx] = a[row * k + tile * TILE_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        if (tile * TILE_SIZE + ty < k && col < n) {
            sharedB[ty][tx] = b[(tile * TILE_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        // wait other threads to finish
        __syncthreads();

        for (int l = 0; l < TILE_SIZE; ++l) {
            sum += sharedA[ty][l] * sharedB[l][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

int main() {
    float *h_a, *h_b, *h_c_naive, *h_c_tile;
    float *d_a, *d_b, *d_c_naive, *d_c_tile;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // allocate memory
    h_a = (float *)malloc(size_A);
    h_b = (float *)malloc(size_B);
    h_c_naive = (float *)malloc(size_C);
    h_c_tile = (float *)malloc(size_C);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c_naive, size_C);
    cudaMalloc(&d_c_tile, size_C);

    // init mat and assign to device
    init_mat(h_a, M, K);
    init_mat(h_b, K, N);
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    // benchmark naive matmul
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    double naive_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double naive_stime = get_time();
        matmul_naive<<<grid_dim, block_dim>>>(d_a, d_b, d_c_naive, M, K, N);
        cudaDeviceSynchronize();
        double naive_etime = get_time();
        naive_time += naive_etime - naive_stime;
    }
    printf("Naive benchmark %f microseconds\n", naive_time / 20.0 * 1000.0);

    // benchmark tile matmul. 
    /*
        we can split result c into tiles, where tile_{i, j} corresponds to tile_a_{i, 0-(N + TILE_SIZE - 1) / TILE_SIZE}
        times tile_b_{(M + TILE_SIZE - 1) / TILE_SIZE, j}. 

        So we can split threads into block of (TILE_SIZE, TILE_SIZE), each thread is responsible for: 
        1. load one scalar of A and B
        2. calculate dot product just like naive

        see tiling.pdf for more information
    */
    dim3 grid_dim_tile((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 block_dim_tile(TILE_SIZE, TILE_SIZE, 1);
    double tile_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        double tile_stime = get_time();
        matmul_tile<<<grid_dim_tile, block_dim_tile>>>(d_a, d_b, d_c_tile, M, K, N);
        cudaDeviceSynchronize();
        double tile_etime = get_time();
        tile_time += tile_etime - tile_stime;
    }
    printf("Tiling benchmark %f microseconds\n", tile_time / 20.0 * 1000.0);

    // check correctness
    cudaMemcpy(h_c_naive, d_c_naive, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_tile, d_c_tile, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (h_c_naive[i] != h_c_tile[i]) {
            correct = false;
            printf("(%d, %d) not equal. h_c_naive: %f, h_c_tile: %f", i / N, i % N, h_c_naive[i], h_c_tile[i]);
            break;
        }
    }
    printf("correctness check: %s\n", correct ? "correct" : "incorrect");

    // free memory
    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_tile);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_tile);
}
