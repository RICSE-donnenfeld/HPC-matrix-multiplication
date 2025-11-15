#!/bin/bash
NPROCS=$(nproc)
export OMP_NUM_THREADS=${1:-$NPROCS}

echo "Building with $OMP_NUM_THREADS threads"

g++ -Ofast -fopenmp -DTYPE=int -include MMpar.h MM_main.cpp -o MMpar
g++ -Ofast -fopenmp -DTYPE=int -include MM.h MM_main.cpp -o MM

