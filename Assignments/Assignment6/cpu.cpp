#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void GameOfLife_CPU(const int* input, int* output, int n) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            int live = 0;
            for (int dr = -1; dr <= 1; dr++){
                for (int dc = -1; dc <= 1; dc++){
                    if (dr == 0 && dc == 0) continue;
                    int nr = (i + dr + n) % n;
                    int nc = (j + dc + n) % n;
                    live += input[nr * n + nc];
                }  
            }  
            int cell = input[i * n + j];
            output[i * n + j] = (cell == 1 && (live == 2 || live == 3)) || (cell == 0 && live == 3) ? 1 : 0;
        }  
    }  
}


void Implemenation_CPU(const int *init, int *output, int n, int iterations){
    size_t size = (size_t)n*n;
    int *a = (int*)malloc(size * sizeof(int));
    int *b = (int*)malloc(size * sizeof(int));
    memcpy(a, init, size * sizeof(int));
    for (int i = 0; i < iterations; i++){
        GameOfLife_CPU(a, b, n);
        int *tmp = a;
        a = b;
        b = tmp;
    }
    memcpy(output, a, size * sizeof(int));
    free(a);
    free(b);
}