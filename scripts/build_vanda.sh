#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

module load gompi/2023a
module load CUDA/12.6.0

mkdir -p bin build

echo "=== Building Serial ==="
g++ -O3 -std=c++17 \
    src/serial_general.cpp \
    -o bin/serial_general

echo "=== Building MPI ==="
mpicxx -O3 -std=c++17 \
    src/mpi_kmeans.cpp \
    -o bin/mpi_kmeans

echo "=== Building Hybrid ==="
mpicxx -O3 -std=c++17 -fopenmp \
    src/hybrid_kmeans.cpp \
    -o bin/hybrid_kmeans

echo "=== Building Optimized Hybrid ==="
mpicxx -O3 -std=c++17 -fopenmp \
    src/hybrid_kmeans_optimized.cpp \
    -o bin/hybrid_kmeans_optimized

echo "=== Building CUDA Object ==="
nvcc -O3 -std=c++17 \
    -c src/cuda_assignment.cu \
    -o build/cuda_assignment.o

echo "=== Building MPI + CUDA ==="
mpicxx -O3 -std=c++17 -fopenmp \
    src/mpi_cuda_kmeans.cpp \
    build/cuda_assignment.o \
    -L"${CUDA_HOME}/lib64" -lcudart -lstdc++ \
    -o bin/mpi_cuda_kmeans

echo "=== BUILD COMPLETED ==="
ls -lh bin/
