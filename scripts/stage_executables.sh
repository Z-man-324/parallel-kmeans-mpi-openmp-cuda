#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/tests"
for program in serial_general mpi_kmeans hybrid_kmeans hybrid_kmeans_optimized mpi_cuda_kmeans; do
  if [[ -f "$ROOT/bin/$program" ]]; then cp "$ROOT/bin/$program" "$ROOT/tests/$program"; fi
done
printf 'PBS executable paths prepared under tests/\n'
