// kdtree_knn.hpp (KD-Tree based k-NN Classifier)

#ifndef KDTREE_KNN_HPP
#define KDTREE_KNN_HPP

#include "kdtree.hpp"
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <omp.h>

// ===============================
// KDTreeKNN Class
// Provides a k-Nearest Neighbors classifier using a KD-Tree as the underlying structure
// ===============================
class KDTreeKNN
{
public:
    KDTree *tree = nullptr; // Pointer to the KDTree instance (dynamically managed)
    int n_neighbors;        // Number of neighbors to consider (k in k-NN)

    // Helper function to determine the most frequent label among nearest neighbors
    int find_most_common(const std::vector<int> &labels) const
    {
        std::unordered_map<int, int> freq;
        int max_count = 0, result = -1;

        for (int label : labels)
        {
            int count = ++freq[label];
            // In case of tie, return smaller label for determinism
            if (count > max_count || (count == max_count && label < result))
            {
                max_count = count;
                result = label;
            }
        }
        return result;
    }

    // Constructor: initializes the classifier with number of neighbors
    KDTreeKNN(int k = 5) : n_neighbors(k)
    {
        if (k <= 0)
            throw std::invalid_argument("k must be positive.");
    }

    // Fit the model by building a KD-Tree from training data
    void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y)
    {
        delete tree; // Clean up existing tree (if any)
        tree = new KDTree(X, y);
    }

    // Predict labels for test data
    std::vector<int> predict(const std::vector<std::vector<double>> &X_test, bool parallel = false) const
    {
        if (!tree)
            throw std::logic_error("Model has not been trained.");

        const size_t n = X_test.size();
        std::vector<int> y_pred;

        if (parallel)
        {
            // ==============================
            // OpenMP Parallel Prediction Loop
            // Each iteration is independent and thread-safe
            // ==============================
            y_pred.resize(n);  // Preallocate space for thread-safe indexed assignment

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(n); ++i)
            {
                auto neighbors = tree->query(X_test[i], n_neighbors);
                y_pred[i] = find_most_common(neighbors);
            }
        }
        else
        {
            // ==============================
            // Sequential Prediction Loop
            // No OpenMP Parallelism
            // ==============================
            y_pred.reserve(n);

            for (const auto &x : X_test)
            {
                auto neighbors = tree->query(x, n_neighbors);
                y_pred.push_back(find_most_common(neighbors));
            }
        }

        return y_pred;
    }


    // Destructor: safely delete KDTree instance
    ~KDTreeKNN()
    {
        delete tree;
    }
};

#endif // KDTREE_KNN_HPP
