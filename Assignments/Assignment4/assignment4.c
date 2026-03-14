#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    // 1. Initialize MPI environment
    MPI_Init(&argc, &argv); 

    double start = MPI_Wtime(); // Start the timer (this is for fun)

    int rank, size; // Size = total number of processes, rank = rank of process
    // 2. Get the rank and size of the processes and store them in MPI variables
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //3. Intitalize bincount, min, max meas, and data count variables
    int bin_count;
    float min_meas, max_meas;
    int data_count;

    // 4. Rank 0 Should take argv (all the arguments) and broadcast them to all other processes
    // First, convert them to appropriate data types
    if (rank == 0) {
        bin_count = atoi(argv[1]);
        min_meas = atof(argv[2]);
        max_meas = atof(argv[3]);
        data_count = atoi(argv[4]);
    }
    // Now we can bvroacast the variables to all processes
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 5. Generate data 

    float *data = NULL; // intialize data pointer to NULL
    if (rank == 0){
        //1. Allocate memory for the data array
        data = (float*)malloc(data_count * sizeof(float)); 
        srand(100); // Set the seed for random number generation to 100 like HW2
        //2. Generate random data between min_meas and max_meas
        for (int i = 0; i < data_count; i++) {
            data[i] = min_meas + ((float)rand() / RAND_MAX) * (max_meas - min_meas);
        }
    }

    // 6. Calculate send counts

    int *send_counts = malloc(size * sizeof(int)); // allocate space for array of send counts
    int *displs = malloc(size * sizeof(int)); // allocate space for array of displacements

    int base = data_count / size; // base number of elements each process will receive
    int remainder = data_count % size; // remaining elements to distribute
    int remaining; // variable to hold the number of remaining elements for each process

    // 7. Calculate send counts and displacements for each process
    for (int i = 0; i < size; i++) {
        // 1. Distribute remainder elements this allows us to do it if the data size is not divisible by the number of processors used.
        if (i < remainder) {
            remaining = 1;
        } else {
            remaining = 0;
        }
        send_counts[i] = base + remaining; // distribute the remainding elements in the remainder

        //2. Calculate displacements
        if (i == 0) {
            displs[i] = 0; // first process starts at index 0
        }else{
            displs[i] = displs[i-1] + send_counts[i-1]; // subsequent processes start after the previous process's data
        }


    }

    // 8. Allocate memory for the local data array on each process
    int local_count;
    // I need to scatter the send counts because only process 0 has the send counts.
    MPI_Scatter(send_counts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD); // Scatter the send counts to all processes
    float *local_data = (float*)malloc(local_count * sizeof(float)); // allocate memory for each thread's buffer to receive the data

    // 9. Scatter the data to all processes

    MPI_Scatterv(data, send_counts, displs, MPI_FLOAT, local_data, local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);


    // 10. each process calculates its local histogram
    // To do this I need a local histogram array of size bin_count initialized to 0
    int *local_histogram = (int*)calloc(bin_count, sizeof(int)); // allocate

    
    // I also need to a way to get the bin index for each data point
    float bin_size = (max_meas - min_meas) / bin_count; // calculate the size of each bin


    // 11. Calculate local histogram for each process

    for (int i = 0; i < local_count; i++){
        // 1. Calculate the bin index for the current data point
        int bin_index = (int)((local_data[i] - min_meas) / bin_size);
        // 2. Handle the case where the data point is exactly max_meas
        if (bin_index >= bin_count) {
            bin_index--;
        }
        // 3. Increment the bin count for the calculated bin index
        local_histogram[bin_index]++;
    }

    // 12. Now that the local histograms are calculated we can send them back to process 0 using mpi reduce with the sum operation 

    // Initialize global histogram on rank 0 which will store the final result
    int *global_histogram = NULL;
    if (rank == 0) {
        global_histogram = (int*)malloc(bin_count * sizeof(int)); // allocate memory for the global histogram on process 0
    }


    MPI_Reduce(local_histogram, global_histogram, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 

    // 13. Compute the bin maxes for printing 
    float *bin_maxes = NULL;
    if (rank == 0) {
        bin_maxes = (float*)malloc(bin_count * sizeof(float));
        for (int i = 0; i < bin_count; i++) {
            bin_maxes[i] = min_meas + (i + 1) * bin_size;
        }
    }

    double end = MPI_Wtime(); // End the timer (this is for fun)

    // 14. Print the final histogram on process 0
    if (rank == 0) {
        printf("bin_maxes: ");
        for (int i = 0; i < bin_count; i++) {
            printf("%.3f ", bin_maxes[i]);
        }
        printf("\nbin_counts: ");
        for (int i = 0; i < bin_count; i++) {
            printf("%d ", global_histogram[i]);
        }
        printf("\n");
        printf("Time taken: %f seconds\n", end - start); // Print the time taken (this is for fun)
    }


    // 15. All done, free the memory

    free(local_data);
    free(local_histogram);
    free(send_counts);
    free(displs);

    if (rank == 0) {
        free(data);
        free(global_histogram);
        free(bin_maxes);

    }


    MPI_Finalize(); // 3. Finalize the MPI environment
    return 0;

}