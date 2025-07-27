// data_utils.hpp

#ifndef DATA_UTILS_HPP
#define DATA_UTILS_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

void load_csv(const std::string& filename,
              std::vector<std::vector<double>>& features,
              std::vector<int>& labels)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::string line;

    // ✅ Read header to count columns
    std::getline(file, line);
    int total_columns = std::count(line.begin(), line.end(), ',') + 1;
    int num_feature_cols = total_columns - 1;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        int label = -1;
        bool row_valid = true;

        for (int col = 0; col < total_columns; ++col) {
            if (!std::getline(ss, cell, ',')) {
                row_valid = false;
                break;
            }

            try {
                if (col < num_feature_cols) {
                    row.push_back(std::stod(cell));
                } else {
                    label = std::stoi(cell);  // Assumes label is an int
                }
            } catch (...) {
                row_valid = false;
                break;
            }
        }

        if (row_valid && row.size() == num_feature_cols) {
            features.push_back(row);
            labels.push_back(label);
        }
    }
    file.close();
}



void generate_random_data(std::vector<std::vector<double>>& features,
                          std::vector<int>& labels,
                          size_t num_samples = 100) 
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> iq_dist(80.0, 160.0);
    std::uniform_real_distribution<double> cgpa_dist(5.0, 10.0);
    std::uniform_int_distribution<int> label_dist(0, 1);

    for (size_t i = 0; i < num_samples; ++i) 
    {
        features.push_back({iq_dist(generator), cgpa_dist(generator)});
        labels.push_back(label_dist(generator));
    }
}

void train_test_split(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_train,
                      std::vector<int>& y_train,
                      std::vector<std::vector<double>>& X_test,
                      std::vector<int>& y_test,
                      double test_size = 0.2) 
{
    if (X.size() != y.size()) 
        throw std::invalid_argument("X and y must be the same length.");

    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    std::mt19937 g(0);  // Fixed seed
    std::shuffle(indices.begin(), indices.end(), g);


    size_t test_count = static_cast<size_t>(X.size() * test_size);
    for (size_t i = 0; i < X.size(); ++i) 
    {
        if (i < test_count) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        } else {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
}
#endif // DATA_UTILS_HPP
