// data_utils.hpp
// Utility functions for CSV data loading, synthetic data generation, and train-test splitting.

#ifndef DATA_UTILS_HPP
#define DATA_UTILS_HPP

#include <fstream>      // For file I/O
#include <sstream>      // For string stream parsing
#include <string>       // For std::string
#include <vector>       // For std::vector
#include <random>       // For random number generation
#include <iostream>     // For debugging/logging
#include <algorithm>    // For std::shuffle, std::count

// ---------------------------------------------------------
// Load CSV file into a feature matrix and label vector
// Assumes last column is the label, rest are features
// ---------------------------------------------------------
void load_csv(const std::string& filename,
              std::vector<std::vector<double>>& features,
              std::vector<int>& labels)
{
    std::ifstream file(filename); // Open file
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::string line;

    // ✅ Read and parse header to determine number of columns
    std::getline(file, line);
    int total_columns = std::count(line.begin(), line.end(), ',') + 1;
    int num_feature_cols = total_columns - 1;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        int label = -1;
        bool row_valid = true;

        // Parse each cell in the row
        for (int col = 0; col < total_columns; ++col) {
            if (!std::getline(ss, cell, ',')) {
                row_valid = false;  // Handle malformed rows
                break;
            }

            try {
                if (col < num_feature_cols) {
                    row.push_back(std::stod(cell)); // Parse feature as double
                } else {
                    label = std::stoi(cell); // Parse label as int
                }
            } catch (...) {
                row_valid = false;  // Handle conversion failure
                break;
            }
        }

        // ✅ Store only if row is valid and has correct number of features
        if (row_valid && row.size() == num_feature_cols) {
            features.push_back(row);
            labels.push_back(label);
        }
    }

    file.close();
}

// ---------------------------------------------------------
// Generate synthetic data for testing or prototyping
// Features: [IQ (80-160), CGPA (5.0-10.0)]
// Labels: Binary (0 or 1)
// ---------------------------------------------------------
void generate_random_data(std::vector<std::vector<double>>& features,
                          std::vector<int>& labels,
                          size_t num_samples = 100) 
{
    std::default_random_engine generator;  // Uses current time or fixed seed
    std::uniform_real_distribution<double> iq_dist(80.0, 160.0);   // Simulate IQ
    std::uniform_real_distribution<double> cgpa_dist(5.0, 10.0);   // Simulate CGPA
    std::uniform_int_distribution<int> label_dist(0, 1);           // Binary class labels

    for (size_t i = 0; i < num_samples; ++i) 
    {
        features.push_back({iq_dist(generator), cgpa_dist(generator)});
        labels.push_back(label_dist(generator));
    }
}

// ---------------------------------------------------------
// Split dataset into training and testing sets
// Randomizes using a fixed seed to ensure reproducibility
// test_size = 0.2 means 20% test, 80% train
// ---------------------------------------------------------
void train_test_split(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_train,
                      std::vector<int>& y_train,
                      std::vector<std::vector<double>>& X_test,
                      std::vector<int>& y_test,
                      double test_size = 0.2) 
{
    // Ensure data and labels have same size
    if (X.size() != y.size()) 
        throw std::invalid_argument("X and y must be the same length.");

    // Create index array to shuffle and split
    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    // Fixed seed ensures same shuffle across runs — useful for reproducibility
    std::mt19937 g(0);  
    std::shuffle(indices.begin(), indices.end(), g);

    // Determine how many samples to allocate to test set
    size_t test_count = static_cast<size_t>(X.size() * test_size);

    // Assign data to test or train based on index position
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
