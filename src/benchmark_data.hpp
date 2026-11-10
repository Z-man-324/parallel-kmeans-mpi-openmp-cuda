#ifndef BENCHMARK_DATA_HPP
#define BENCHMARK_DATA_HPP

#include <vector>
#include <random>
#include <stdexcept>
#include <string>
#include <cstddef>

inline std::vector<double> generate_benchmark_data(
    int N,
    int D,
    int K,
    unsigned int seed,
    const std::string& mode
) {
    if (N <= 0 || D <= 0 || K <= 0 || K > N) {
        throw std::invalid_argument("Invalid N, D, or K");
    }

    if (mode != "easy" && mode != "overlap") {
        throw std::invalid_argument("Unknown dataset mode");
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);

    const double spacing =
        (mode == "easy") ? 10.0 : 1.0;

    std::vector<double> data(
        static_cast<std::size_t>(N) * D
    );

    for (int i = 0; i < N; ++i) {
        int cluster = i % K;

        for (int d = 0; d < D; ++d) {
            double center = spacing * cluster;

            data[static_cast<std::size_t>(i) * D + d] =
                center + noise(rng);
        }
    }

    return data;
}

#endif
