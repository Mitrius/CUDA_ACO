#!/bin/bash
module load mpi/mpich-x86_64
mpicxx \-o main.out src/main.cpp
chmod +x ./main.out