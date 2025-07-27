// main.cpp
// Modify ➜ mingw32-make ➜ .\knn.exe


#include "data_utils.hpp"    // For load_csv and train_test_split utilities
#include "knn.hpp"           // Custom KNN class implementation
#include "evaluate.hpp"      // For accuracy_score function
#include <chrono>            // For measuring prediction time

int main()
{
    std::vector<std::vector<double>> features; // Matrix to hold feature vectors (each row = one sample)
    std::vector<int> labels;                   // Corresponding class labels

    // Load preprocessed CSV dataset: each row = [features..., label]
    // Modify path accordingly if running on another system
    load_csv("C:\\Users\\Dell\\Desktop\\myWorkPlace\\PROJECTS\\KNN C++\\data\\fashion_combined.csv", features, labels);

    // Split data into training and testing sets
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(features, labels, X_train, y_train, X_test, y_test, 0.2); // 80% training, 20% test

    // Initialize KNN with k = 6 neighbors (hyperparameter)
    KNN knn(6);

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

    // Last output:
    // Prediction time: 362.029 seconds  --> Long time likely due to brute-force distance computation on a large dataset
    // Accuracy: 83.8419%                --> Acceptable result, depending on dataset complexity and separability
}
