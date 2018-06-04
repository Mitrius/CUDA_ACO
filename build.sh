#!/bin/bash
#module load mpi/mpich-x86_64
mpicc -c main.cpp -o main.o
nvcc -c kernel.cu -o kernel.o
mpicc main.o kernel.o -lcudart -o main.out
chmod +x ./main.out