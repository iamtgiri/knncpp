// knn_cli_main.cpp

#include "data_utils.hpp"
#include "evaluate.hpp"
#include "knn.hpp"
#include "kdtree_knn.hpp"
#include <cxxopts.hpp>
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    cxxopts::Options options("KNN-Benchmark", "k-NN Benchmarking using Brute Force or KD-Tree with optional OpenMP");

    options.add_options()("d,data", "Path to dataset CSV file", cxxopts::value<std::string>()->default_value("C:\\Users\\Dell\\Desktop\\myWorkPlace\\PROJECTS\\KNN C++\\data\\fashion_combined.csv"))
    ("r,test_size", "Train/Test Split Ratio (0.0 - 1.0)", cxxopts::value<float>()->default_value("0.2"))
    ("k,k_neighbors", "Number of Neighbors", cxxopts::value<int>()->default_value("8"))
    ("m,mode", "Algorithm Mode: brute or kdtree", cxxopts::value<std::string>()->default_value("brute"))
    ("p,parallel", "Enable Parallel Prediction (OpenMP)", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string data_path = result["data"].as<std::string>();
    float train_ratio = result["test_size"].as<float>();
    int k = result["k_neighbors"].as<int>();
    std::string mode = result["mode"].as<std::string>();
    bool parallel = result["parallel"].as<bool>();

    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    load_csv(data_path, features, labels);
    std::cout << "[INFO] Loaded data with " << features.size() << " samples.\n";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(features, labels, X_train, y_train, X_test, y_test, train_ratio);

    std::vector<int> y_pred;
    auto start = std::chrono::high_resolution_clock::now();

    // Select and execute prediction mode
    if (mode == "brute" || mode == "kdtree")
    {
        std::cout << "[INFO] Using " << (mode == "brute" ? "Brute-force" : "KD-Tree") 
                << " KNN with k = " << k << ".\n";

        if (parallel)
            std::cout << "[INFO] OpenMP parallel prediction enabled.\n";
        else
            std::cout << "[INFO] Using sequential prediction (no OpenMP).\n";

        if (mode == "brute")
        {
            KNN knn(k);
            knn.fit(X_train, y_train);
            y_pred = knn.predict(X_test, parallel);
        }
        else  // mode == "kdtree"
        {
            KDTreeKNN knn(k);
            knn.fit(X_train, y_train);
            y_pred = knn.predict(X_test, parallel);
        }
    }
    else
    {
        std::cerr << "[ERROR] Invalid mode specified. Use 'brute' or 'kdtree'.\n";
        return 1;
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double accuracy = accuracy_score(y_test, y_pred) * 100;

    std::cout << "Prediction time: " << duration.count() << " seconds\n";
    std::cout << "Accuracy: " << accuracy << "%\n";

    return 0;
}
