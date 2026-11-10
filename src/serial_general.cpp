#include "kmeans_common.hpp"
#include "benchmark_data.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include <cmath>

struct KMeansResult {
    Dataset centroids;
    int iterations;
    double inertia;
};

Dataset generate_data(int N, int D, int K, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);

    Dataset data(N, Point(D));

    for (int i = 0; i < N; ++i) {
        int cluster = i % K;

        for (int d = 0; d < D; ++d) {
            double center = 10.0 * cluster;
            data[i][d] = center + noise(rng);
        }
    }

    return data;
}

KMeansResult run_kmeans(
    const Dataset& data,
    Dataset centroids,
    int max_iterations = 100,
    double tolerance = 1e-6
) {
    const int N = static_cast<int>(data.size());
    const int D = static_cast<int>(data[0].size());

    const int K = static_cast<int>(centroids.size());

    int iterations = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Dataset sums(K, Point(D, 0.0));
        std::vector<int> counts(K, 0);

        // Assignment and local accumulation
        for (int i = 0; i < N; ++i) {
            int best_cluster = 0;
            double best_distance =
                std::numeric_limits<double>::max();

            for (int k = 0; k < K; ++k) {
                double dist =
                    squared_distance(data[i], centroids[k]);

                if (dist < best_distance) {
                    best_distance = dist;
                    best_cluster = k;
                }
            }

            counts[best_cluster]++;

            for (int d = 0; d < D; ++d) {
                sums[best_cluster][d] += data[i][d];
            }
        }

        // Update centroids
        double max_shift = 0.0;

        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) {
                continue;
            }

            for (int d = 0; d < D; ++d) {
                double new_value =
                    sums[k][d] / counts[k];

                double diff =
                    new_value - centroids[k][d];

                max_shift += diff * diff;

                centroids[k][d] = new_value;
            }
        }

        iterations = iter + 1;

        if (max_shift < tolerance * tolerance) {
            break;
        }
    }

    // Compute final inertia
    double inertia = 0.0;

    for (int i = 0; i < N; ++i) {
        double best_distance =
            std::numeric_limits<double>::max();

        for (int k = 0; k < K; ++k) {
            double dist =
                squared_distance(data[i], centroids[k]);

            if (dist < best_distance) {
                best_distance = dist;
            }
        }

        inertia += best_distance;
    }

    return {centroids, iterations, inertia};
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " N D K Seed\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int D = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    unsigned int seed =
        static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10));

    if (N <= 0 || D <= 0 || K <= 0 || K > N) {
        std::cerr << "Invalid parameters\n";
        return 1;
    }

    std::string mode = (argc == 6) ? argv[5] : "easy";

    std::vector<double> flat_data =
        generate_benchmark_data(N, D, K, seed, mode);

    Dataset data(N, Point(D));

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            data[i][d] =
                flat_data[static_cast<std::size_t>(i) * D + d];
        }
    }

    // Initialize centroids before timing.
    Dataset initial_centroids(K, Point(D));

    for (int k = 0; k < K; ++k) {
        initial_centroids[k] = data[k];
    }

    auto start = std::chrono::steady_clock::now();

    KMeansResult result = run_kmeans(data, initial_centroids);

    auto end = std::chrono::steady_clock::now();

    double runtime =
        std::chrono::duration<double>(end - start).count();

    std::cout << std::setprecision(12);
    std::cout << "N=" << N
              << " D=" << D
              << " K=" << K
              << " Seed=" << seed << "\n";

    std::cout << "Iterations: "
              << result.iterations << "\n";

    std::cout << "Inertia: "
              << result.inertia << "\n";

    std::cout << "Runtime (seconds): "
              << runtime << "\n";

    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ":";

        for (double value : result.centroids[k]) {
            std::cout << " " << value;
        }

        std::cout << "\n";
    }

    return 0;
}
