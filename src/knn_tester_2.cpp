// knn_tester_2.cpp

#include "data_utils.hpp"
#include "knn.hpp"
#include "evaluate.hpp"
#include <chrono>

int main()
{
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    load_csv("C:\\Users\\Dell\\Desktop\\myWorkPlace\\PROJECTS\\KNN C++\\data\\Social_Network_Ads_knn_data.csv", features, labels);

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    train_test_split(features, labels, X_train, y_train, X_test, y_test, 0.2);

    KNN knn(6);
    knn.fit(X_train, y_train);

    auto start = std::chrono::high_resolution_clock::now();
    auto y_pred = knn.predict(X_test);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Prediction time: " << duration.count() << " seconds\n";

    // std::cout << "\nPredictions:\n";
    // for (size_t i = 0; i < y_pred.size(); ++i)
    // {
    //     std::cout << "Sample " << i
    //               << " -> Predicted: " << y_pred[i]
    //               << ", Actual: " << y_test[i]
    //               << " [";
    //     for (size_t j = 0; j < X_test[i].size(); ++j)
    //     {
    //         std::cout << X_test[i][j];
    //         if (j < X_test[i].size() - 1)
    //             std::cout << ", ";
    //     }
    //     std::cout << "]\n";
    // }

    std::cout << "Accuracy: " << accuracy_score(y_test, y_pred) * 100 << "%\n";

    return 0;
}
