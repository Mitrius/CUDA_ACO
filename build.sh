#!/bin/bash
#module load mpi/mpich-x86_64
mpicc -c src/main.cpp -o main.o -std=c++11
nvcc -c src/kernel.cu -o kernel.o
mpicc main.o kernel.o -L/usr/local/cuda/lib64 -lcudart -lcurand -o main.out
chmod +x ./main.out
