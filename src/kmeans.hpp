// kmeans.hpp

#ifndef KMEANS_HPP
#define KMEANS_HPP
#include "vector.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>

class KMeans
{
public:
    int n_clusters, max_iter;
    std::vector<Vector> centroids = {};
    double inertia_ = 0.0;

    KMeans(int n_clusters = 2, int max_iter = 200) : n_clusters(n_clusters), max_iter(max_iter) {}

    std::vector<int> fit_predict(const std::vector<std::vector<double>> &X)
    {
        // Randomly initialize centroids
        srand(time(NULL));
        for (int i = 0; i < n_clusters; ++i)
        {
            centroids.push_back(Vector(X[std::rand() % X.size()]));
        }

        std::vector<int> cluster_group = assign_clusters(X);
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Assign clusters
            cluster_group = assign_clusters(X);
            std::vector<Vector> old_centroids = centroids;
            // move centroids
            centroids = move_centroids(X, cluster_group);
            inertia_ = calculate_inertia(X, cluster_group);
            // check finish
            if (std::equal(centroids.begin(), centroids.end(), old_centroids.begin()))
            {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }
        return cluster_group;
    }
    std::vector<int> assign_clusters(const std::vector<std::vector<double>> &X)
    {
        std::vector<int> cluster_group;
        for (const auto &row : X)
        {
            std::vector<double> distance;
            for (const auto &centroid : centroids)
            {
                distance.push_back(Vector(row).euclideanDistance(centroid));
            }
            int min_index = std::min_element(distance.begin(), distance.end()) - distance.begin();
            cluster_group.push_back(min_index);
        }
        return cluster_group;
    }

    std::vector<Vector> move_centroids(const std::vector<std::vector<double>> &X, const std::vector<int> &cluster_group)
    {
        std::vector<Vector> new_centroids(n_clusters, Vector(X[0].size()));
        std::vector<int> counts(n_clusters, 0);

        for (size_t i = 0; i < X.size(); ++i)
        {
            new_centroids[cluster_group[i]] = new_centroids[cluster_group[i]] + Vector(X[i]);
            counts[cluster_group[i]]++;
        }

        for (int i = 0; i < n_clusters; ++i)
        {
            if (counts[i] > 0)
            {
                new_centroids[i] = new_centroids[i] / counts[i];
            }
        }
        return new_centroids;
    }

    double calculate_inertia(const std::vector<std::vector<double>> &X, const std::vector<int> &cluster_group)
    {
        double inertia = 0.0;
        for (size_t i = 0; i < X.size(); ++i)
        {
            inertia += Vector(X[i]).euclideanDistance(centroids[cluster_group[i]]);
        }
        return inertia;
    }
};
#endif // KMEANS_HPP