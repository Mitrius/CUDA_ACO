#!/bin/bash
#module load mpi/mpich-x86_64
mpic++ -c src/main.cpp -o main.o -std=c++11
nvcc -c src/kernel.cu -o kernel.o
mpic++ kernel.o main.o -std=c++11 -L/usr/local/cuda/lib64 -o main.out -lcudart -lcurand
chmod +x ./main.out
