/*
 * This is the basics of the basics, but to be honest, pretty hard to start. I will explain the hierarchy of a
 * CUDA kernel to an extent that is beginner-friendly
 *
 * Workflow of a CUDA project
 * 1. CPU copies the data into GPU
 * 2. GPU did all the intensive calculation tasks
 * 3. GPU write the result back to CPU
 *
 * High level understanding:
 * You can think of CPU as the boss, and GPU is a bunch of skilled workers
 * CPU assigns a large tasks to GPU (basically sends the data to GPU)
 * GPU workers, since they are skilled, can figure out how to split the task and each work get one of the sub tasks.
 * When all the workers finish their task, they integrate them and send back to CPU
 *
 * CUDA kernel hierarchy
 * GPU is all about managing threads, and executing them in parallel to reach a super fast speed.
 * In a CUDA kernel, one basically needs to do two things:
 * 1. Organize the required threads in grid-block structure
 * 2. Implement the instructions that will be applied to all the threads of this kernel
 *
 * Organizing threads:
 * step1:   CUDA programmers need to first figure out how many threads they want to create
 * step2:   Then they group threads into blocks. For example, if there are a total of 10000 threads, they can group
 *          the threads into 100 blocks, where each block has 100 threads. We can either set the block dims as 1D(1 by 100)
 *          2D(10 * 10, etc), or 3D(5*5*4, etc).
 * There are two reasons for this design:
 * 1.   the first is related to nvcc(compiler) and GPU hardware, as this structure helps nvcc to generate efficient machine code
 *      such that GPU hardware can be utilized with higher efficiency
 * 2.   One important mindset in CUDA programming is that you need to use thread id to access the data this thread requires
 *      By such a design, we can easily align the structure of what we need to the actual structure of the data stored in memory
 * We will have tons of example to build such mindset
 *
 * Implementation:
 * In this part, we tell GPU what exactly is each thread doing. What we usually need to do is extracting the threadidx and
 * blockidx to obtain the actual index in the original data. You may feel strange that these variables are like coming out of
 * no where. You can consider it as for granted, and will be set up during compilation stage.
 *
 *
 */

 #include <stdio.h>

 /*
     This function prints out the idx of current thread
     We have 4 built-in variables:
     girdDim: dimension of the grid
     blockDim: dimension of the block
     blockIdx: which block it is in the grid
     ThreadIdx: which thread it is in the block

     One really important thing to remember is:
     for both block and grid:
     .x -> col
     .y -> row
     .z -> depth

     the following code shows a flattening of the threads
     However, in many situations, we don't have to flatten the grid into a list, we can have 2D index or even 3D index, making 
     it easier to compute global index of each thread. (Check vector_add.cu and matmul.cu)

     To acquire deeper understanding of indexing, the only way is to practice!
 */
 
 __global__ void whoami(void) {
     int block_id =
             blockIdx.x +  
             blockIdx.y * gridDim.x +    // floor number in this building (rows high)
             blockIdx.z * gridDim.x * gridDim.y;   // room number
 
     int block_offset =
             block_id * // times our apartment number
             blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)
 
     int thread_offset =
             threadIdx.x +
             threadIdx.y * blockDim.x +
             threadIdx.z * blockDim.x * blockDim.y;
 
     int id = block_offset + thread_offset; // global person id in the entire apartment complex
 
     printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
            id,
            blockIdx.x, blockIdx.y, blockIdx.z, block_id,
            threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
     // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
 }
 
 int main() {
     const int b_x = 2, b_y = 3, b_z = 4; // 24 blocks inside a grid
     const int t_x = 4, t_y = 4, t_z = 4; // 64 threads in a block
 
     int block_per_grid = b_x * b_y * b_z;
     int thread_per_block = t_x * t_y * t_z;
 
     printf("blocks per grid: %d\n", block_per_grid);
     printf("threads per block: %d\n", thread_per_block);
     printf("total threads: %d\n", thread_per_block * block_per_grid);
 
     // dim3 is a useful helper type for stating the grid and block dimension (grid dimnsion is block per grid, and block dimension is thread per grid)
     /*
        CUDA convention will set x as col, y as row and z as depth
        So, your actual dimension in the following is (b_z, b_y, b_x)
        This is important when you are writing your instructions in your kernel
     */
     dim3 blockPerGrid(b_x, b_y, b_z);
     dim3 threadPerBlock(t_x, t_y, t_z);
 
     // This is the format for calling thread function
     whoami<<<blockPerGrid, threadPerBlock>>>();
 
     // Wait until all threads are finished
     cudaDeviceSynchronize();
     printf("finished\n");
 }
