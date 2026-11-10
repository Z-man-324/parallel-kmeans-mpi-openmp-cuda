#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error: %s\n",                        \
                cudaGetErrorString(err));                          \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

__global__ void assign_points_kernel(
    const double* data,
    const double* centroids,
    int* assignments,
    int N,
    int D,
    int K
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    int best_cluster = 0;
    double best_distance = 1.7976931348623157e308;

    for (int k = 0; k < K; ++k) {
        double dist = 0.0;

        for (int d = 0; d < D; ++d) {
            double diff =
                data[i * D + d] - centroids[k * D + d];

            dist += diff * diff;
        }

        if (dist < best_distance) {
            best_distance = dist;
            best_cluster = k;
        }
    }

    assignments[i] = best_cluster;
}

extern "C" void cuda_assign_points(
    const double* host_data,
    const double* host_centroids,
    int* host_assignments,
    int N,
    int D,
    int K
) {
    double *device_data = nullptr;
    double *device_centroids = nullptr;
    int *device_assignments = nullptr;

    size_t data_bytes = (size_t)N * D * sizeof(double);
    size_t centroid_bytes = (size_t)K * D * sizeof(double);
    size_t assignment_bytes = (size_t)N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&device_centroids, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&device_assignments, assignment_bytes));

    CUDA_CHECK(cudaMemcpy(
        device_data, host_data,
        data_bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        device_centroids, host_centroids,
        centroid_bytes, cudaMemcpyHostToDevice
    ));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    assign_points_kernel<<<blocks, threads>>>(
        device_data,
        device_centroids,
        device_assignments,
        N, D, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        host_assignments,
        device_assignments,
        assignment_bytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFree(device_centroids));
    CUDA_CHECK(cudaFree(device_assignments));
}
