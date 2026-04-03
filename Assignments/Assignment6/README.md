# Assignment 5.2 - Conway's Game of Life

## By Ammon Phipps

## Instructions on running code

1. Compile on cluster

    - run `module load cuda`
    - run `nvcc -O3 -arch=native -std=c++14 -o life main.cu`
        - if this fails you'll have to specify architecture manually.
          - `nvcc -O3 -arch=sm_80 -std=c++14 -o life main.cu` for example, depending on GPU

2. Run
    - Request a GPU node
      - I used a GTX Titan on Kings Peak Cluster
    - run `./life [optional input file name] (defaults to "gc_1024x1024-uint8.raw")`

3. Visualize
    - run `python3 visualize.py`