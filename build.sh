#!/bin/bash
#module load mpi/mpich-x86_64
mpicc -c src/main.cpp -o main.o
nvcc -c src/kernel.cu -o kernel.o
mpicc main.o kernel.o -lcudart -o main.out
chmod +x ./main.out