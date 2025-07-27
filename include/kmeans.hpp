// kmeans.hpp

#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "vector.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>

// KMeans clustering algorithm implementation
class KMeans
{
public:
    int n_clusters, max_iter;
    std::vector<Vector> centroids = {}; // Stores current centroids
    double inertia_ = 0.0;              // Sum of distances of points to their closest centroid (used to evaluate fit)

    // Constructor initializes number of clusters and maximum iterations
    KMeans(int n_clusters = 2, int max_iter = 200) : n_clusters(n_clusters), max_iter(max_iter) {}

    // Fit the model to data and return cluster assignments
    std::vector<int> fit_predict(const std::vector<std::vector<double>> &X)
    {
        // === STEP 1: Random initialization of centroids ===
        srand(time(NULL)); // Use current time as random seed
        for (int i = 0; i < n_clusters; ++i)
        {
            centroids.push_back(Vector(X[std::rand() % X.size()])); // Pick random data points as initial centroids
        }

        // === STEP 2: Initial cluster assignment ===
        std::vector<int> cluster_group = assign_clusters(X);

        // === STEP 3: Iteratively update centroids and assignments ===
        for (int iter = 0; iter < max_iter; ++iter)
        {
            cluster_group = assign_clusters(X); // Assign data points to closest centroid
            std::vector<Vector> old_centroids = centroids;

            centroids = move_centroids(X, cluster_group); // Move centroids to mean of assigned points
            inertia_ = calculate_inertia(X, cluster_group); // Evaluate current configuration

            // === Check for convergence ===
            if (std::equal(centroids.begin(), centroids.end(), old_centroids.begin()))
            {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }

        return cluster_group;
    }

    // Assign each data point to the nearest centroid
    std::vector<int> assign_clusters(const std::vector<std::vector<double>> &X)
    {
        std::vector<int> cluster_group;

        for (const auto &row : X)
        {
            std::vector<double> distance;

            // Compute distance from this point to each centroid
            for (const auto &centroid : centroids)
            {
                distance.push_back(Vector(row).euclideanDistance(centroid));
            }

            // Assign to cluster with minimum distance
            int min_index = std::min_element(distance.begin(), distance.end()) - distance.begin();
            cluster_group.push_back(min_index);
        }

        return cluster_group;
    }

    // Move centroids to the mean of their assigned points
    std::vector<Vector> move_centroids(const std::vector<std::vector<double>> &X, const std::vector<int> &cluster_group)
    {
        std::vector<Vector> new_centroids(n_clusters, Vector(X[0].size())); // Initialize zero vectors for new centroids
        std::vector<int> counts(n_clusters, 0); // Count of points assigned to each cluster

        // Sum all points belonging to the same cluster
        for (size_t i = 0; i < X.size(); ++i)
        {
            new_centroids[cluster_group[i]] = new_centroids[cluster_group[i]] + Vector(X[i]);
            counts[cluster_group[i]]++;
        }

        // Divide by count to get mean (i.e., new centroid)
        for (int i = 0; i < n_clusters; ++i)
        {
            if (counts[i] > 0)
            {
                new_centroids[i] = new_centroids[i] / counts[i];
            }
        }

        return new_centroids;
    }

    // Calculate total inertia (sum of distances to assigned centroids)
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
