#include <mpi.h>
#include "benchmark_data.hpp"
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Insufficient MPI thread support\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5 && argc != 6) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " N D K Seed\n";
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int K = atoi(argv[3]);
    unsigned int seed =
        static_cast<unsigned int>(strtoul(argv[4], nullptr, 10));

    if (N <= 0 || D <= 0 || K <= 0 || K > N) {
        if (rank == 0)
            cerr << "Invalid parameters\n";
        MPI_Finalize();
        return 1;
    }

    std::string mode = (argc == 6) ? argv[5] : "easy";

    // Rank 0 generates the same data as the serial baseline.
    vector<double> full_data;
    vector<double> centroids(K * D, 0.0);

    if (rank == 0) {
        full_data = generate_benchmark_data(
            N, D, K, seed, mode
        );

        // Same deterministic initialization as serial.
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d)
                centroids[k * D + d] =
                    full_data[static_cast<size_t>(k) * D + d];
    }

    // Partition points as evenly as possible.
    vector<int> sendcounts(size), displs(size);
    int offset = 0;

    for (int r = 0; r < size; ++r) {
        int local_n = N / size + (r < N % size ? 1 : 0);
        sendcounts[r] = local_n * D;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    int local_n = sendcounts[rank] / D;
    vector<double> local_data(sendcounts[rank]);

    MPI_Scatterv(
        rank == 0 ? full_data.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_data.data(), sendcounts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Bcast(
        centroids.data(), K * D, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    const int max_iterations = 100;
    const double tolerance = 1e-6;
    int iterations = 0;

    int actual_threads = 0;

    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        cout << "Actual OpenMP threads per rank: "
             << actual_threads << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Allocate reusable buffers once, outside the iteration loop.
    const int threads = omp_get_max_threads();

    vector<double> local_sums(K * D, 0.0);
    vector<int> local_counts(K, 0);

    vector<vector<double>> thread_sums(
        threads, vector<double>(K * D, 0.0));
    vector<vector<int>> thread_counts(
        threads, vector<int>(K, 0));

    vector<double> global_sums(K * D, 0.0);
    vector<int> global_counts(K, 0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        fill(local_sums.begin(), local_sums.end(), 0.0);
        fill(local_counts.begin(), local_counts.end(), 0);

        for (int t = 0; t < threads; ++t) {
            fill(thread_sums[t].begin(), thread_sums[t].end(), 0.0);
            fill(thread_counts[t].begin(), thread_counts[t].end(), 0);
        }

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            vector<double>& sums = thread_sums[tid];
            vector<int>& counts = thread_counts[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < local_n; ++i) {
                int best_cluster = 0;
                double best_distance =
                    numeric_limits<double>::max();

                for (int k = 0; k < K; ++k) {
                    double dist = 0.0;
                    for (int d = 0; d < D; ++d) {
                        double diff =
                            local_data[i * D + d] -
                            centroids[k * D + d];
                        dist += diff * diff;
                    }

                    if (dist < best_distance) {
                        best_distance = dist;
                        best_cluster = k;
                    }
                }

                counts[best_cluster]++;

                for (int d = 0; d < D; ++d) {
                    sums[best_cluster * D + d] +=
                        local_data[i * D + d];
                }
            }
        }

        // Merge thread-local statistics within each MPI rank.
        for (int t = 0; t < threads; ++t) {
            for (int k = 0; k < K; ++k)
                local_counts[k] += thread_counts[t][k];

            for (int j = 0; j < K * D; ++j)
                local_sums[j] += thread_sums[t][j];
        }

        MPI_Allreduce(
            local_sums.data(), global_sums.data(),
            K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
        );

        MPI_Allreduce(
            local_counts.data(), global_counts.data(),
            K, MPI_INT, MPI_SUM, MPI_COMM_WORLD
        );

        double total_shift = 0.0;

        for (int k = 0; k < K; ++k) {
            if (global_counts[k] == 0)
                continue;

            for (int d = 0; d < D; ++d) {
                double new_value =
                    global_sums[k * D + d] / global_counts[k];

                double diff = new_value - centroids[k * D + d];
                total_shift += diff * diff;
                centroids[k * D + d] = new_value;
            }
        }

        iterations = iter + 1;

        if (total_shift < tolerance * tolerance)
            break;
    }

    // Final inertia using the final centroids.
    double local_inertia = 0.0;

    for (int i = 0; i < local_n; ++i) {
        double best_distance = numeric_limits<double>::max();

        for (int k = 0; k < K; ++k) {
            double dist = 0.0;

            for (int d = 0; d < D; ++d) {
                double diff =
                    local_data[i * D + d] - centroids[k * D + d];
                dist += diff * diff;
            }

            best_distance = min(best_distance, dist);
        }

        local_inertia += best_distance;
    }

    double global_inertia = 0.0;

    MPI_Reduce(
        &local_inertia, &global_inertia,
        1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
    );

    double local_runtime = MPI_Wtime() - start;
    double runtime = 0.0;

    MPI_Reduce(
        &local_runtime, &runtime,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        cout << setprecision(12);
        cout << "N=" << N << " D=" << D
             << " K=" << K << " Seed=" << seed << "\n";
        cout << "MPI ranks: " << size << "\n";
        cout << "Iterations: " << iterations << "\n";
        cout << "Inertia: " << global_inertia << "\n";
        cout << "Runtime (seconds): " << runtime << "\n";

        for (int k = 0; k < K; ++k) {
            cout << "Centroid " << k << ":";
            for (int d = 0; d < D; ++d)
                cout << " " << centroids[k * D + d];
            cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
