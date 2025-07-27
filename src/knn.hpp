// knn.hpp

#ifndef KNN_HPP
#define KNN_HPP
#include "vector.hpp"
#include <algorithm>
#include <unordered_map>

class KNN
{
public:
    int n_neighbors;
    std::vector<Vector> X_train;
    std::vector<int> y_train;
    KNN(int n_neighbors = 5) : n_neighbors(n_neighbors)
    {
        if (n_neighbors <= 0)
            throw std::invalid_argument("Number of neighbors must be > 0.");
    }

    void fit(const std::vector<std::vector<double>> &X_train, const std::vector<int> &y_train)
    {
        if (X_train.empty() || y_train.empty() || X_train.size() != y_train.size())
            throw std::invalid_argument("Invalid training data.");

        this->X_train.clear();
        this->y_train.clear();
        for (const auto &row : X_train)
        {
            this->X_train.push_back(Vector(row));
        }
        this->y_train = y_train;
    }

    std::vector<int> predict(const std::vector<std::vector<double>> &X_test)
    {
        if (X_test.empty())
            throw std::invalid_argument("Empty test data.");
        std::vector<int> y_pred;
        for (const auto &test_row : X_test)
        {
            std::vector<std::pair<double, int>> indexed_distances;
            Vector test_vec(test_row);
            indexed_distances.reserve(X_train.size());
            for (int i = 0; i < X_train.size(); ++i)
            {
                // double dist = test_vec.euclideanDistance(X_train[i]);
                double dist = test_vec.euclideanSquaredDistance(X_train[i]);
                indexed_distances.emplace_back(dist, i);
            }
            // std::sort(indexed_distances.begin(), indexed_distances.end());
            std::partial_sort(
                indexed_distances.begin(),
                indexed_distances.begin() + n_neighbors,
                indexed_distances.end());

            std::vector<int> nearest_labels;
            for (int i = 0; i < n_neighbors && i < indexed_distances.size(); ++i)
            {
                nearest_labels.push_back(y_train[indexed_distances[i].second]);
            }
            int most_common_label = find_most_common(nearest_labels);
            y_pred.push_back(most_common_label);
        }
        return y_pred;
    }
    int find_most_common(const std::vector<int> &labels)
    {
        std::unordered_map<int, int> count_map;
        for (const auto &label : labels)
        {
            ++count_map[label];
        }
        int most_common = labels[0];
        int max_count = 0;
        for (const auto &pair : count_map)
        {
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