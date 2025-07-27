// knn.hpp

#ifndef KNN_HPP
#define KNN_HPP

#include "vector.hpp"         // Custom Vector class used for distance calculation
#include <algorithm>          // For sorting/presorting distances
#include <unordered_map>      // For frequency count in majority voting

// K-Nearest Neighbors classifier
class KNN
{
public:
    int n_neighbors;                        // Number of neighbors to consider (hyperparameter)
    std::vector<Vector> X_train;           // Training feature vectors, stored as custom Vector objects
    std::vector<int> y_train;              // Corresponding training labels

    // Constructor with default k=5; throws exception if invalid k is passed
    KNN(int n_neighbors = 5) : n_neighbors(n_neighbors)
    {
        if (n_neighbors <= 0)
            throw std::invalid_argument("Number of neighbors must be > 0.");
    }

    // Stores the training dataset into internal member variables
    void fit(const std::vector<std::vector<double>> &X_train, const std::vector<int> &y_train)
    {
        // Sanity check for input consistency
        if (X_train.empty() || y_train.empty() || X_train.size() != y_train.size())
            throw std::invalid_argument("Invalid training data.");

        this->X_train.clear();
        this->y_train.clear();

        // Convert each row into a Vector for easy distance operations
        for (const auto &row : X_train)
        {
            this->X_train.push_back(Vector(row));
        }
        this->y_train = y_train;
    }

    // Predicts class labels for a given test dataset
    std::vector<int> predict(const std::vector<std::vector<double>> &X_test)
    {
        if (X_test.empty())
            throw std::invalid_argument("Empty test data.");

        std::vector<int> y_pred;  // Vector to store final predictions

        for (const auto &test_row : X_test)
        {
            std::vector<std::pair<double, int>> indexed_distances; // Pair of (distance, index in training data)
            Vector test_vec(test_row); // Wrap test row into a Vector object
            indexed_distances.reserve(X_train.size());

            // Compute squared Euclidean distances from test_vec to all training points
            for (int i = 0; i < X_train.size(); ++i)
            {
                double dist = test_vec.euclideanSquaredDistance(X_train[i]); // Faster than true distance, since sqrt not needed
                indexed_distances.emplace_back(dist, i);
            }

            // Retrieve the k smallest distances using partial sort (more efficient than full sort)
            std::partial_sort(
                indexed_distances.begin(),
                indexed_distances.begin() + n_neighbors,
                indexed_distances.end());

            // Extract the labels of the k nearest neighbors
            std::vector<int> nearest_labels;
            for (int i = 0; i < n_neighbors && i < indexed_distances.size(); ++i)
            {
                nearest_labels.push_back(y_train[indexed_distances[i].second]);
            }

            // Assign the majority class label from the neighbors
            int most_common_label = find_most_common(nearest_labels);
            y_pred.push_back(most_common_label);
        }

        return y_pred;
    }

    // Returns the most frequently occurring label among the k neighbors
    int find_most_common(const std::vector<int> &labels)
    {
        std::unordered_map<int, int> count_map; // Frequency map: label -> count

        // Count occurrences of each label
        for (const auto &label : labels)
        {
            ++count_map[label];
        }

        // Determine the most common label
        int most_common = labels[0];
        int max_count = 0;
        for (const auto &pair : count_map)
        {
            // Tie-breaking: prefer the smaller label in case of equal frequency
            if (pair.second > max_count || (pair.second == max_count && pair.first < most_common))
            {
                max_count = pair.second;
                most_common = pair.first;
            }
        }

        return most_common;
    }
};

#endif // KNN_HPP
