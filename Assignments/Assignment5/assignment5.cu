// I compiled this program like this:

// module load cuda
// nvcc -o rgb2grey assignment5.cu

// On kingspeak.chpc.utah.edu


#include <stdio.h>
#include <stdlib.h>

// Define the width and height of the image our constants
#define WIDTH 1024
#define HEIGHT 1024


// THIS IS THE KERNEL FUNCTION TO CONVERT RGB TO GRAYSCALE
// RUN BY THE GPU
__global__ void rgbToGrayscale(unsigned char *d_rgb, unsigned char *d_gray, int width, int height){
    // 1. Calculate the thread Index for col and row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    // 2. Check if the thread is within the bounds of the image
    if (x < width && y < height) {
        // 3. Calculate the index for the RGB and Grayscale arrays
        int rgbIndex = (y * width + x) * 3; // Each pixel has 3 components (R, G, B)
        int grayIndex = y * width + x; // Each pixel has 1 component (Grayscale)
        // 4. Get the RGB values from the input array
        unsigned char R = d_rgb[rgbIndex];     // Red component
        unsigned char G = d_rgb[rgbIndex + 1]; // Green component
        unsigned char B = d_rgb[rgbIndex + 2]; // Blue component
        // 5. Convert RGB to Grayscale 
        d_gray[grayIndex] = static_cast<unsigned char>(0.299f * R + 0.587f * G + 0.114f * B);

    
    } 
}


int main(int argc, char *argv[]){
    // 1. Define our variables
    // total number of pixels, rbg size (numpixels * 3), and gray size (numpixels)
    // Part 2: blockSizes array for different block sizes to test part 2 of assignment
    int numPixels = WIDTH * HEIGHT;
    int rgbSize = numPixels * 3;
    int graySize = numPixels;
    int blockSizes[] = {8, 16, 32}; // 64 blocks, 256 blocks, 1024 blocks

    // 2. Allocate memory for the RGB and Grayscale images on the HOST
    unsigned char *h_rgb = (unsigned char *)malloc(rgbSize * sizeof(unsigned char));
    unsigned char *h_gray = (unsigned char *)malloc(graySize * sizeof(unsigned char));

    // 3. Read the input file

    FILE *inputFile = fopen("gc_conv_1024x1024.raw" , "rb");
        // Read RGB data from file and store it in h_rgb
    fread(h_rgb, 1, rgbSize, inputFile); 
    fclose(inputFile);

    // 4. Allocate memory for the RGB and Grayscale images on the GPU (DEVICE)
    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void **)&d_rgb, rgbSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_gray, graySize * sizeof(unsigned char));

    // 5. Copy the RGB image from the HOST to the DEVICE
    cudaMemcpy(d_rgb, h_rgb, rgbSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //6. Launch the kernel

        // 6a. Define the block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y); // enough blocks to cover the entire image

        // 6b. Call the kernel function to convert RGB to Grayscale
    rgbToGrayscale<<<gridDim, blockDim>>>(d_rgb, d_gray, WIDTH, HEIGHT);

    // 7. Copy the Grayscale image from the DEVICE back to the HOST
    cudaMemcpy(h_gray, d_gray, graySize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 8. Write the Grayscale image to an output file
    FILE *outputFile = fopen("gc.raw", "wb");
    fwrite(h_gray, 1, graySize, outputFile);
    fclose(outputFile);


    ///////////////////////////////////////////////////////
    // ASSIGNMENT 5.1 PART 2: TESTING DIFFERENT BLOCK SIZES
    ///////////////////////////////////////////////////////

    for(int i = 0; i< 3; i++){
        //1. define block and grid size
        dim3 block(blockSizes[i], blockSizes[i]); // block size from the array
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y); // grid size based on the block size
        // 2. create events, record start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // 3. launch the kernel 200 times
        for (int j = 0; j<200; j++){
            rgbToGrayscale<<<grid, block>>>(d_rgb, d_gray, WIDTH, HEIGHT);
        }
        // 4. record stop time, calculate elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // 5. print the block size and elapsed time
        printf("Block Size: %dx%d - TOTAL Time for 200 runs: %.2f ms\n", blockSizes[i], blockSizes[i], milliseconds);

    }

    // Free the allocated memory on both the HOST and DEVICE
    free(h_rgb);
    free(h_gray);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    return 0;
}

//////////////////////////////////////////////
// ASSIGNMENT 5.1 PART 2 ANALYSIS
//////////////////////////////////////////////
/*
The gpu I am using is a Nvidia GeForce GTX TITAN X, here are its specs

Max Threads per SM: 2048
Max Blocks per SM: 8


So, I am going to test block sizes of 8x8 (64 threads), 16x16 (256 threads), and 32x32 (1024 threads) to see how the performance changes with different block sizes.

1. Block Size 8x8:

    8 blocks *  64 threads = 512 threads per sm 
    This is well below the max number of threads per sm, because of this
    We will not be using the GPU to its full potential and I think it will have
    the worse performance. 
    512/2048 = 25%

2. Block Size 16x16:

    8 blocks * 256 threads = 2048 threads per sm
    This is the maximum number of threads per sm, so we will be fully utilizing the GPU's capabilities.
    I think it will have much better performance than the 8x8 block size because we are at 100%
    2048/2048 = 100%

3. Block Size 32x32:

    2 blocks * 1024 threads = 2048 threads per sm
    This is also the maximum number of threads per sm, so we will also be fully utilizing the GPU's capabilities.
    I think it will have similar performance to the 16x16 block size, but slightly worse because we have fewer blocks
    2048/2048 = 100%

===========================================================================================================================
                                                    RESULTS


This the output of my program after running it on the Nvidia GeForce GTX TITAN X GPU:

    ```
    [u6074058@kp297:~]$ ./rgb2grey
    Block Size: 8x8 - TOTAL Time for ALL 200: 11.57 ms
    Block Size: 16x16 - TOTAL Time for ALL 200: 5.66 ms
    Block Size: 32x32 - TOTAL Time for ALL 200: 6.41 ms
    ```

    8x8 was the slowest
    16x16 was the fastest
    32x32 was slightly slower than 16x16

*/

