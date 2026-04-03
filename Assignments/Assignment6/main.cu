#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Import 3 types of implementations: CPU, GPU naive, and GPU tiled and utils

#include "cpu.cpp"
#include "gpu_naive.cu"
// #include "gpu_tiled.cu"

// #include "utils.cpp"




#define N 1024
#define ITERATIONS 1000

int main(int argc, char **argv){
    // 1. Get the input file from cmd or use default
    const char *input = (argc > 1) ? argv[1] : "gc_1024x1024-uint8.raw";

    int n = N;
    size_t size = (size_t)n*n;

    // 2. Load the grid from the file and create input array
    int *h_init = (int*)malloc(size * sizeof(int));
    FILE *f = fopen(input, "rb");

    if (!f){
        fprintf(stderr, "Error opening file %s\n", input);
        return 1;
    }

    // Read raw bytes then expand to int
    uint8_t *tmp = (uint8_t*)malloc(size);
    if (fread(tmp, 1, size, f) != size){
        fprintf(stderr, "Error reading file %s\n", input);
        return 1;
    }
    fclose(f);

    for (size_t i = 0; i < size; i++) h_init[i] = (int)tmp[i];
    free(tmp);

    printf("Loaded grid from %s\n", input);

    // 3. Create output array
    int *h_out = (int*)malloc(size * sizeof(int));

    // 4. Run CPU implementation

    printf("----CPU SERIAL IMPLEMENTATION----\n");
    // 4.1 Create output array for CPU result
    int *cpu_result = (int*)malloc(size * sizeof(int));

    struct timespec start, end;

    // 4.2 Run the CPU implementation and measure time

    clock_gettime(CLOCK_MONOTONIC, &start);
    Implemenation_CPU(h_init, cpu_result, n, ITERATIONS);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double cpu_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;

    // 4.2 Print CPU Report
    printf("CPU Time: %.2f ms\n", cpu_ms);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Grid Size: %dx%d\n", n, n);
    
    // 4.3 Save CPU output to file
    FILE *f_cpu = fopen("cpu_output.raw", "wb");
    if (!f_cpu) {
        fprintf(stderr, "Error opening file cpu_output.raw\n");
        return 1;
    }
    fwrite(cpu_result, sizeof(int), size, f_cpu);
    fclose(f_cpu);

    printf("Output file saved to cpu_output.raw\n");

    // 5. Run GPU Naive implementation

    printf("----GPU NAIVE IMPLEMENTATION----\n");
    
    float gpu_naive_ms;
    Implementation_GPU_Naive(h_init, h_out, n, ITERATIONS, &gpu_naive_ms);
    // 5.1 Print GPU Naive Report

    printf("GPU Naive Time: %.2f ms\n", gpu_naive_ms);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Grid Size: %dx%d\n", n, n);
    float timesFasterThanCPU = cpu_ms / gpu_naive_ms;
    printf("GPU Naive is %.2f times faster than CPU\n", timesFasterThanCPU);


    // 5.2 Save GPU Naive output to file
    FILE *f_gpu = fopen("gpu_naive_output.raw", "wb");
    if (!f_gpu) {
        fprintf(stderr, "Error opening file gpu_naive_output.raw\n");
        return 1;
    }
    fwrite(h_out, sizeof(int), size, f_gpu);
    fclose(f_gpu);

    printf("Output file saved to gpu_naive_output.raw\n");

    printf("----GPU TILED IMPLEMENTATION----\n");
    // 6. Run GPU Tiled implementation
    float gpu_tiled_ms;
    Implementation_GPU_Tiled(h_init, h_out, n, ITERATIONS, &gpu_tiled_ms);
    // 6.1 Print GPU Tiled Report
    printf("GPU Tiled Time: %.2f ms\n", gpu_tiled_ms);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Grid Size: %dx%d\n", n, n);
    timesFasterThanCPU = cpu_ms / gpu_tiled_ms;
    printf("GPU Tiled is %.2f times faster than CPU\n", timesFasterThanCPU);

    // 6.2 Save GPU Tiled output to file
    FILE *f_gpu_tiled = fopen("gpu_tiled_output.raw", "wb");
    if (!f_gpu_tiled) {
        fprintf(stderr, "Error opening file gpu_tiled_output.raw\n");
        return 1;
    }
    fwrite(h_out, sizeof(int), size, f_gpu_tiled);
    fclose(f_gpu_tiled);
    

    // free resources
    free(h_init);
    free(h_out);
    free(cpu_result);


    // You are done

    printf("COMPLETE\nAll implementations ran successfully.\n");



    return 0;

}

