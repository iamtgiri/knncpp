// knn_main.cpp (Brute-force kNN implementation)

#include "data_utils.hpp" // For load_csv and train_test_split utilities
#include "knn.hpp"        // Custom KNN class implementation
#include "evaluate.hpp"   // For accuracy_score function
#include <chrono>         // For measuring prediction time

int main()
{
    std::vector<std::vector<double>> features; // Matrix to hold feature vectors (each row = one sample)
    std::vector<int> labels;                   // Corresponding class labels

    // Load preprocessed CSV dataset: each row = [features..., label]
    // Modify path accordingly if running on another system
    load_csv("C:\\Users\\Dell\\Desktop\\myWorkPlace\\PROJECTS\\KNN C++\\data\\fashion_combined.csv", features, labels);
    std::cout << "[INFO] Loaded data with " << features.size() << " samples\n";

    // Split data into training and testing sets
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(features, labels, X_train, y_train, X_test, y_test, 0.2); // 80% training, 20% test

    // Initialize KNN with k = 8 neighbors (hyperparameter)
    int n_neighbors = 8;
    KNN knn(n_neighbors=n_neighbors);
    std::cout << "[INFO] Training KNN with k=" << n_neighbors << " neighbors\n";
    std::cout << "[INFO] Training data size: " << X_train.size() << "samples.\n";
    std::cout << "[INFO] Test data size: " << X_test.size() << " samples.\n";
    std::cout << "[INFO] Starting KNN prediction...\n";
    // Train KNN by storing training data (lazy learning - no actual model fitting)
    knn.fit(X_train, y_train);

    // Measure how long prediction takes (important for large datasets)
    auto start = std::chrono::high_resolution_clock::now();
    auto y_pred = knn.predict(X_test); // Predict labels for test data
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Prediction time: " << duration.count() << " seconds\n";

    // Evaluate performance using accuracy metric
    std::cout << "Accuracy Score: " << accuracy_score(y_test, y_pred) * 100 << "%\n";

    // ========== Brute-force KNN Performance Analysis ==========
    //
    // Baseline (No OpenMP parallelism):
    //   - Prediction Time: 362.029 seconds
    //     → High latency due to exhaustive pairwise distance computations on a large dataset.
    //   - Accuracy: 84.042%
    //     → Validates correct implementation; performance acceptable given dataset complexity.
    //
    // Optimized (With OpenMP parallelism):
    //   - Prediction Time: 6.46803 seconds
    //     → ~55x speedup achieved via parallelized distance computation using OpenMP.
    //   - Accuracy: 84.042%
    //     → Identical to the serial version, confirming correctness and numerical stability of parallel logic.
}
