// tester.cpp

#include "vector.hpp"
#include "kmeans.hpp"
#include <iostream>

int main()
{
    // Toy dataset: [IQ, CGPA]
    std::vector<std::vector<double>> data = {
        {120, 8.5}, {130, 9.1}, {115, 7.8}, {140, 9.5}, {110, 7.0},
        {100, 6.5}, {105, 6.8}, {95, 6.0}, {102, 6.2}, {108, 6.9},
        {150, 9.8}, {145, 9.3}, {98, 5.8}, {113, 7.4}, {125, 8.2},
        {90, 5.5}, {88, 5.2}, {132, 9.0}, {138, 9.2}, {85, 5.0}
    };

    // Create KMeans instance
    int k = 3;  // You can test with other k
    KMeans km(k);
    
    // Perform clustering
    std::vector<int> cluster_labels = km.fit_predict(data);

    // Output results
    std::cout << "\nCluster assignments:\n";
    for (size_t i = 0; i < data.size(); ++i)
    {
        std::cout << "Sample " << i << " -> Cluster " << cluster_labels[i]
                  << " [IQ: " << data[i][0] << ", CGPA: " << data[i][1] << "]\n";
    }

    std::cout << "\nFinal inertia: " << km.inertia_ << std::endl;

    return 0;
}
