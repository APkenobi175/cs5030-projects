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




__global__ void GameOfLife_GPU_Naive(const int *input, int *output, int n){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;

    int live = 0;
    for (int dr = -1; dr <= 1; dr++){
        for (int dc = -1; dc <= 1; dc++){       
            if (dr == 0 && dc == 0) continue;
            int nr = (r + dr + n) % n;
            int nc = (c + dc + n) % n;
            live += input[nr * n + nc];
        }
    }
    int cell = input[r * n + c];
    output[r * n + c] = (cell == 1 && (live == 2 || live == 3)) || (cell == 0 && live == 3) ? 1 : 0;
}


void Implementation_GPU_Naive(const int *h_init, int *h_out, int n, int iterations, float *out_ms){
    size_t size = (size_t)n*n*sizeof(int);

    // 1. Allocate device memory
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    // 2. Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_init, size, cudaMemcpyHostToDevice));
    // 3. Define block and grid sizes

    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (n + 31) / 32);

    // 4. Create CUDA events for timing, and start recording
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // 5. Run the kernel for the specified number of iterations
    for (int i = 0; i < iterations; i++){
        GameOfLife_GPU_Naive<<<grid, block>>>(d_input, d_output, n);
        int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    // 6. Stop recording, calculate elapsed time, copy result back to host
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(out_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_input, size, cudaMemcpyDeviceToHost));

    // 7. Clean up

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    
}