#ifndef KMEANS_COMMON_HPP
#define KMEANS_COMMON_HPP

#include <vector>
#include <cmath>
#include <cstddef>

using Point = std::vector<double>;
using Dataset = std::vector<Point>;

inline double squared_distance(
    const Point& a,
    const Point& b
) {
    double sum = 0.0;

    for (std::size_t d = 0; d < a.size(); ++d) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }

    return sum;
}

#endif
