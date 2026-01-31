// // Write (and upload) a program using Pthreads or C++11 threads that implements the histogram program discussed in "4 - ParallelSoftware". 
// The program will have too:
// populate an array (data) of <data_count> float elements between <min_meas> and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.
// compute the histogram (i.e., bin_maxes and bin_count) using  <number of threads> threads using a global sum
// compute the histogram (i.e., bin_maxes and bin_count) using  <number of threads> threads using a tree structured sum
// The inputs of the program are:

// <number of threads>, the number of threads to use for the execution
// <bin_count>, the number of bins in the histogram
// <min_meas>, minimum (float) value of the measurements
// <max_meas>, maximum (float) value of the measurements
// <data_count>, number of measurements
// Your program must adhere to the following order of command-line arguments:
// <number of threads> <bin count> <min meas> <max meas> <data count>


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Important: note that for the same input and varying number of threads the output must be the same. 
// Discuss the reasons why results might differ for different executions (or number of threads) and what 
// technique did you implemented to solve this problem.
/////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Results can differ between executions with different number of threads for a few reasons.
// 1. They are not synchronized, so threads are executing at different times, leading to them getting done out of order
// 2. Floating point math isn't associative so the order of operations can change the final result slightly
// 3. To solve this problem, I made sure that each thread only writes to its own local histogram 
//    and only combines the results at the end. THus, no race conditions can occur and the order of operations 
//    is consistent regardless of the number of threads used.
/////////////////////////////////////////////////////////////////////////////////////////////////////




#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <tuple> // for std::tuple
#include <cstdlib> // for exit, and rand/srand
#include <algorithm> // for std::min
#include <mutex> // for std::mutex
#include <condition_variable> // for std::condition_variable






// Used to create on the fly debug print statements
void debug_print(std::string msg){
    std::cout << msg << std::endl;
}

// Print command usage information to the console
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <number of threads> <bin count> <min meas> <max meas> <data count>" << std::endl;
}

std::tuple<int, int, float, float, int> parseArgs(int argc, char* argv[]){
// This function parses command line arguments and returns them as appropriate types in a tuple

    //0. Check if the number of arguments is correct
    if (argc != 6) {
        printUsage(argv[0]);
        exit(EXIT_FAILURE);
    }
    // 1. Convert the command line arguments to appropriate types
    int thread_count = std::stoi(argv[1]);
    int bin_count = std::stoi(argv[2]);
    float min_meas = std::stof(argv[3]);
    float max_meas = std::stof(argv[4]);
    int data_count = std::stoi(argv[5]);

    // DEBUG
    //debug_print("Parsed Arguments:");
    //debug_print("Thread Count: " + std::to_string(thread_count));
    //debug_print("Bin Count: " + std::to_string(bin_count));
    //debug_print("Min Measurement: " + std::to_string(min_meas));
    //debug_print("Max Measurement: " + std::to_string(max_meas));
    //debug_print("Data Count: " + std::to_string(data_count));

    // 2. If you get here that means all the arguments were able to be converted successfully
    // 3. Make sure they are valid
    if(thread_count <= 0){
        std::cout << "Invalid thread count: " << thread_count << std::endl;
        exit(EXIT_FAILURE);
    }
    if(bin_count <= 0){
        std::cout << "Invalid bin count: " << bin_count << std::endl;
        exit(EXIT_FAILURE);
    }
    if(min_meas >= max_meas){
        std::cout << "Invalid measurement range: min_meas (" << min_meas << ") >= max_meas (" << max_meas << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(data_count <= 0){
        std::cout << "Invalid data count: " << data_count << std::endl;
        exit(EXIT_FAILURE);
    }


    // 4.If all arguments are valid, return the parsed arguments as a tuple

    return std::tuple<int, int, float, float, int>(thread_count, bin_count, min_meas, max_meas, data_count); // return a tuple of parsed arguments

}

std::vector<float> getData(int data_count, float min_meas, float max_meas){
    // This function generates an array of float measurements between min_meas and max_meas
    
    //1. Create a vector of size data_count to hold the measurements
    std::vector<float> data(data_count);
    //2. Calculate the range for the random values
    float range = max_meas - min_meas;
    //3. For each element in the vector, generate a random float between min_meas and max_meas
    for(int i = 0; i<data_count; i++){
        float random_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i] = min_meas + random_value * range; // scale it to the desired range
    }
    return data;
}


// These 2 Functions are helper functions for histogram computing

static std::vector<float> computeBinMaxes(int bin_count, float min_meas, float max_meas){
    // This function computes the maximum value for each bin in the histogram
    //1. Create a vector to hold the bin maxes
    std::vector<float> bin_maxes(bin_count);
    // 2. Calculate the width of each bin
    float bin_width = (max_meas - min_meas) / bin_count;
    //3 . For each bin, calculate its maximum value with the formula: min_meas + (i + 1) * bin_width
    for(int i = 0; i < bin_count; i++){
        bin_maxes[i] = min_meas + (i + 1) * bin_width;
    }
    return bin_maxes;
}

int getBinIndex(float x, float min_meas, float max_meas, int bin_count){
    // 1. Calculate the width of each bin
    float bin_width = (max_meas - min_meas) / bin_count;
    // 2. Determine the bin index for the measurement x
    int bin_index = static_cast<int>((x - min_meas) / bin_width); // this truncates towards zero, so it's like floor for positive numbers
    // 3. Ensure the bin index is within valid range
    if(bin_index < 0) bin_index = 0;
    if(bin_index >= bin_count) bin_index = bin_count - 1;

    // debug_print("Measurement: " + std::to_string(x) + " assigned to Bin Index: " + std::to_string(bin_index));

    return bin_index;
}




std::vector<int> globalSumHistogram(const std::vector<float>& data,
                                   int bin_count, float min_meas, float max_meas,
                                   int thread_count) {

    // This function computes the histogram using a global sum approach with multiple threads
    // Each thread will write only to its own local histogram to avoid race conditions
    // After all threads complete, the local histograms will be added together to form the final histogram

    //1. Create a 2D vector to hold local histograms for each thread
    std::vector<std::vector<int>> locals(thread_count, std::vector<int>(bin_count, 0));

    //2. Create and launch threads to compute local histograms
    std::vector<std::thread> threads;
    threads.reserve(thread_count); 


    // 3. Determine the chunk size for each thread
    int n = static_cast<int>(data.size()); // remember data is the input vector of measurements, so n is the total number of elements
    int chunk = (n + thread_count - 1) / thread_count; // all elements divided by number of threads, rounded up



    // 4. For each thread, assign a chunk of data so that start is threads i * chunk and end is min(n, start + chunk)
    for (int i = 0; i < thread_count; i++) {
        int start = i * chunk;
        int end = std::min(n, start + chunk);


        // 5. Launch the thread to compute its local histogram
        // This uses a lambda function for the thread's work
        // It then saves the result in a local histogram
        threads.emplace_back([&, i, start, end]() { // lambda function for the thread, each thread gets its own i, start, end
            for (int j = start; j < end; j++) {
                int b = getBinIndex(data[j], min_meas, max_meas, bin_count);
                locals[i][b]++;  
            }
        });
    }

    // 6. Wait for all threads to finish building their local histograms
    for (auto& th : threads) th.join();

    // 7. Combine local histograms into the final histogram
    std::vector<int> bin_counts(bin_count, 0);
    for (int i = 0; i < thread_count; i++) {
        for (int b = 0; b < bin_count; b++) {
            bin_counts[b] += locals[i][b];
        }
    }


    // 8. Return the final histogram
    return bin_counts;
}




// create a barrier struct to hold each thread's results
struct Barrier {
    std::vector<int> counts; 
    bool ready = false; 
    std::mutex mtx; 
    std::condition_variable cv;
};

std::vector<int> treeSumHistogram(const std::vector<float>& data, int bin_count, float min_meas, float max_meas, int thread_count) {
    int n = data.size();
    int chunk = (n + thread_count - 1) / thread_count;

    // 1. Create a vector of barrier structs to hold each thread's results
    std::vector<Barrier> results(thread_count);

    // 2. Initialize each thread's counts vector
    for(int i = 0; i < thread_count; ++i) {
        results[i].counts.assign(bin_count, 0);
    }


    // 3. Create and launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_count; i++) {
        threads.emplace_back([&, i]() {
            int start = i * chunk;
            int end = std::min(n, start + chunk);

            // 4. Local Histogram Computation
            for (int j = start; j < end; j++) {
                int b = getBinIndex(data[j], min_meas, max_meas, bin_count);
                results[i].counts[b]++;
            }

            // 5. Now that the histogram is completed, we will add the results using a tree structured method
            for (int step = 1; step < thread_count; step *= 2) {
                // Determine if the thread is a reciever or a sender
                if (i % (2 * step) == 0) {
                    int partner = i + step;
                    if (partner < thread_count) {
                        // Wait for partner thread to be ready
                        std::unique_lock<std::mutex> lock(results[partner].mtx);
                        results[partner].cv.wait(lock, [&] { return results[partner].ready; });

                        // Whent he partner is ready, add its counts to current thread's counts
                        for (int b = 0; b < bin_count; b++) {
                            results[i].counts[b] += results[partner].counts[b];
                        }
                    }
                } 
                // If the thread is a sender
                else if (i % (2 * step) == step) {
                    // Signal to partner thread that its ready
                    {
                        std::lock_guard<std::mutex> lock(results[i].mtx);
                        results[i].ready = true;
                    }
                    // Notify the partner thread
                    results[i].cv.notify_one();
                    break; // Sender no longer needed
                }
            }
        });
    }

    // 6. Wait for all threads to finish

    for (auto& th : threads) th.join();

    // 7. The final histogram is in results[0]
    return results[0].counts;
}



int main(int argc, char* argv[]){
    srand(100); // Initialize random seed


    // 1. Parse command line arguments
    std::tuple<int, int, float, float, int> args = parseArgs(argc, argv);
    int thread_count = std::get<0>(args);
    int bin_count = std::get<1>(args);
    float min_meas = std::get<2>(args);
    float max_meas = std::get<3>(args);
    int data_count = std::get<4>(args);


    // 2. Generate data
    std::vector<float> data = getData(data_count, min_meas, max_meas);

    //3. Compute bin maxes
    std::vector<float> bin_maxes = computeBinMaxes(bin_count, min_meas, max_meas);

    //4. Compute histogram using global sum
    //debug_print("Computing histogram using global sum...");
    std::vector<int> global_histogram = globalSumHistogram(data, bin_count, min_meas, max_meas, thread_count);
    //debug_print("If you get here that means it worked!");
    
    //5. Compute histogram using tree structured sum
    //debug_print("Computing histogram using tree structured sum...");
    std::vector<int> tree_histogram = treeSumHistogram(data, bin_count, min_meas, max_meas, thread_count);
    //debug_print("If you get here that means it worked!");


    //6. Print results
    std::cout << "Global Sum:\n";
    std::cout << "Bin Maxes: ";
    for (const auto& max : bin_maxes) {
        std::cout << max << " ";
    }
    std::cout << "\nBin Counts: ";
    for (const auto& count : global_histogram) {
        std::cout << count << " ";
    }

    std::cout << "\n\nTree Structured Sum:\n";
    std::cout << "Bin Maxes: ";
    for (const auto& max : bin_maxes) {
        std::cout << max << " ";
    }
    std::cout << "\nBin Counts: ";
    for (const auto& count : tree_histogram) {
        std::cout << count << " ";
    }
    std::cout << std::endl;

}




