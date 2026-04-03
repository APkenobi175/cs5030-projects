#include <stdio.h>
#include <cuda_runtime.h>

// 1. Cuda error checking
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n",                 \
              __FILE__, __LINE__, cudaGetErrorString(err));                  \
      exit(EXIT_FAILURE);                                                    \
    }                                                                       \
  } while(0)

#define TILE_W 32
#define TILE_H 32
#define HALO 1


__global__ void GameOfLife_GPU_Tiled(const int *input, int *output, int n){

    // 1. Calculate global row and column indices
    // This is our little cache that we make
    __shared__ int tile[TILE_H + 2 * HALO][TILE_W + 2*HALO];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gc = blockIdx.x * TILE_W + tx; // Global column
    int gr = blockIdx.y * TILE_H + ty; // Global row

    int sx = tx + HALO; // Shared memory column index
    int sy = ty + HALO; // Shared memory row index

    // 2. Load the tile into shared memory, including halo cells
    #define FETCH(row, col) input[((row + n) % n) * n + ((col + n) % n)]

    tile[sy][sx] = FETCH(gr, gc); // Center cell

    // 3. Load Edges

    if (tx == 0) tile[sy][0] = FETCH(gr, gc - 1); // Left halo
    if (tx == TILE_W - 1) tile[sy][TILE_W + 1] = FETCH(gr, gc + 1); // Right halo
    if (ty == 0) tile[0][sx] = FETCH(gr - 1, gc); // Top halo
    if (ty == TILE_H - 1) tile[TILE_H + 1][sx] = FETCH(gr + 1, gc); // Bottom halo

    // 4. Load Corners

    if (tx == 0 && ty == 0) tile[0][0] = FETCH(gr - 1, gc - 1); // Top-left
    if (tx == TILE_W - 1 && ty == 0) tile[0][TILE_W + 1] = FETCH(gr - 1, gc + 1); // Top-right
    if (tx == 0 && ty == TILE_H - 1) tile[TILE_H + 1][0] = FETCH(gr + 1, gc - 1); // Bottom-left
    if (tx == TILE_W - 1 && ty == TILE_H - 1) tile[TILE_H + 1][TILE_W + 1] = FETCH(gr + 1, gc + 1); // Bottom-right

    #undef FETCH

    __syncthreads(); // Ensure all threads have loaded their data

    if (gr >= n || gc >= n) return; // Boundary check

    // 5 Now that we have the tile in shared memory, we can compute thhe game of life rules for our cell

    int live = 0;
    for (int dr = -1; dr <= 1; dr++){
        for (int dc = -1; dc <= 1; dc++){
            if (dr == 0 && dc == 0) continue; // Skip the center cell
            live += tile[sy + dr][sx + dc];
        }
    }
    int cell = tile[sy][sx];
    output[gr * n + gc] = (cell == 1 && (live == 2 || live == 3)) || (cell == 0 && live == 3) ? 1 : 0;
}

void Implementation_GPU_Tiled(const int *h_init, int *h_out, int n, int iterations, float *out_ms){

    size_t size = (size_t)n*n*sizeof(int);

    // 1. Allocate device memory

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_init, size, cudaMemcpyHostToDevice));
    // 2. Define block and grid sizes
    dim3 block(TILE_W, TILE_H);
    dim3 grid((n + TILE_W - 1) / TILE_W, (n + TILE_H - 1) / TILE_H);

    // 3. Create CUDA events for timing, and start recording
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // 4. Run the kernel for the specified number of iterations
    for (int i = 0; i < iterations; i++){
        GameOfLife_GPU_Tiled<<<grid, block>>>(d_input, d_output, n);
        int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    // 5. Stop recording, calculate elapsed time, copy result back to host
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(out_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_input, size, cudaMemcpyDeviceToHost));

    // 6. Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    
}


