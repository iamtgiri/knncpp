#pragma once
#include <string>
#include <iostream>

struct CLIOptions {
    std::string data =
        "E:\\myWorkPlace\\PROJECTS\\KNN C++\\data\\fashion_combined.csv";
    float test_size = 0.2f;
    int k_neighbors = 8;
    std::string mode = "brute";
    bool parallel = false;
    bool help = false;
};

inline void print_help() {
    std::cout <<
        "KNN-Benchmark\n"
        "Usage:\n"
        "  knn [options]\n\n"
        "Options:\n"
        "  -d, --data <path>        Path to dataset CSV\n"
        "  -r, --test_size <float> Train/Test split ratio (0-1)\n"
        "  -k, --k_neighbors <int> Number of neighbors\n"
        "  -m, --mode <brute|kdtree>\n"
        "  -p, --parallel           Enable OpenMP\n"
        "  -h, --help               Show this help\n";
}
