
/*
Assignment 3 Questions - Ammon Phipps

Compiled using
    gcc -g -Wall -fopenmp -O2 -o count_sort Assignment3.c
Run using
    ./count_sort <number of threads> <number of elements>

1. If we try to parallelize the for i loop (the outer loop), which variables should be private and which should be shared?

    - the shared variables need to be a, the original array, n, the size of the array, and temp, the temporary array. 
    - The private variable needs to be count so that each thread has its own count variable when it does count++;
    - since we declare count inside the for loop, its automatically private so we don't need to declare it in our openMP pragma or else we get a compilation error....

2. If we consider the memcpy implementation NOT thread safe, how would you approach parralelizing this operation?

    - We could use another for loop and do it the old fashioned why by iterating through each element and setting a[i] = temp[i].
    - Then we can add another openMP pragma for this loop as well.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void print_array(const char* name, int* array, int size){
    // Helper function to print an array with a name
    printf("%s: ", name);
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void count_sort_omp(int a[], int n, int threads){
    int* temp = malloc(n * sizeof(int)); // create an array to hold the input values
    if (!temp){
        fprintf(stderr, "Memory allocation failed!!!\n");
        exit(1);
    }


    #pragma omp parallel for num_threads(threads) default(none) shared(a, n, temp)
    // Declare a, n, and temp as shared variables, and count as a private variable (since it's declared inside the loop)
    for (int i = 0; i < n; i++){
        int count = 0;
        for (int j = 0; j < n; j++){
            if (a[j] < a[i]){
                count ++;
            }
            else if (a[j] == a[i] && j < i){
                count++;
            }
        }
        temp[count] = a[i]; // place the element in its correct position in the temp array
    }
    // memcpy(a, temp, n*sizeof(int)); // copy the sorted elements array (temp) back to the original array, a
    // Parralelize the above memcpy using OpenMP
    #pragma omp parallel for num_threads(threads) default(none) shared(a, n, temp)
    // Declare a, n, and temp as shared variables, since we're just reading from temp and writing to a, we don't need any private variables
    // We are just iterating through temp and setting its values to a.
    for (int i = 0; i < n; i++){
        a[i] = temp[i]; // copy the sorted elements array (temp) back to the original array, a
    }
    free(temp); // free the memory allocated for the temp array
}


int main(int argc, char* argv[]){
    srand(100); 
    if (argc != 3){
        fprintf(stderr, "Usage: %s <number of threads> <number of elements>\n", argv[0]);
        return 1;
    }

    // get num threads and num elements from command line args

    int n = atoi(argv[2]);
    int threads = atoi(argv[1]);

    int* a = malloc(n * sizeof(int)); // create an array to hold the input values
    if (!a){
        fprintf(stderr, "Memory allocation failed!!!\n");
        return 1;
    }

    // Fill the array with random integers
    for (int i = 0; i < n; i++){
        a[i] = rand() % 100; // random integers between 0 and 99
    }

    print_array("Original array", a, n);
    count_sort_omp(a, n, threads);
    print_array("Sorted array", a, n);

    free(a); // free the memory allocated for the input array
    return 0;
}