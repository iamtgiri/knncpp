// kdtree_knn_main.cpp (Using KD-Tree for kNN)

#include "data_utils.hpp"
#include "evaluate.hpp"
#include "kdtree_knn.hpp"

#include <chrono>
#include <iostream>

int main() {
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    load_csv("C:\\Users\\Dell\\Desktop\\myWorkPlace\\PROJECTS\\KNN C++\\data\\fashion_combined.csv", features, labels);
    std::cout << "[INFO] Loaded " << features.size() << " samples.\n";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(features, labels, X_train, y_train, X_test, y_test);

    int n_neighbors = 8;
    KDTreeKNN knn(n_neighbors=n_neighbors);
    std::cout << "[INFO] Fitting KDTree-based kNN...\n";
    knn.fit(X_train, y_train);

    std::cout << "[INFO] Predicting...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto y_pred = knn.predict(X_test);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Prediction time: " << duration.count() << " seconds\n";
    std::cout << "Accuracy: " << accuracy_score(y_test, y_pred) * 100 << "%\n";


    // ========== KD-Tree KNN Performance Analysis ==========
    //
    // Baseline (No OpenMP parallelism):
    //   - Prediction Time: 18.9928 seconds
    //     → Moderate latency due to sequential nearest-neighbor search across all test samples.
    //     → Each test point requires log(N)-depth recursive search, but processed serially.
    //   - Accuracy: 84.042%
    //     → Confirms functional correctness of the KD-Tree and kNN voting implementation.
    //
    // Optimized (With OpenMP parallelism):
    //   - Prediction Time: 7.52332 seconds
    //     → ~2.5x speedup achieved by parallelizing inference over test samples using OpenMP.
    //     → Thread-safe design ensured by preallocating result vector and avoiding shared writes.
    //   - Accuracy: 84.042%
    //     → Identical to the serial version, demonstrating deterministic, stable behavior under multithreading.
    //     → Confirms thread safety of KD-Tree query logic and kNN label aggregation.

    return 0;
}
