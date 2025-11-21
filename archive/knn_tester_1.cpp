// knn_tester_1.cpp


/* RUN
        g++ src/knn_tester.cpp -o knn_app
        ./knn_app
*/


#include <iostream>
#include "knn.hpp"
#include "evaluate.hpp"


int main()
{
    try
    {
        // Example dataset (IQ, CGPA), class labels: 0 = average, 1 = excellent
        std::vector<std::vector<double>> X_train = {
            {120, 8.5}, {130, 9.1}, {100, 6.5},
            {110, 7.0}, {140, 9.5}, {105, 6.8},
            {95, 6.0},  {150, 9.8}, {85, 5.0},
            {115, 7.8}
        };

        std::vector<int> y_train = {
            1, 1, 0,
            0, 1, 0,
            0, 1, 0,
            1
        };

        std::vector<std::vector<double>> X_test = {
            {125, 8.0}, {90, 5.5}, {135, 9.3}
        };

        std::vector<int> y_expected = {1, 0, 1}; // Just for comparison

        // Create and train classifier
        KNN knn(3);  // k = 3
        knn.fit(X_train, y_train);

        // Predict
        std::vector<int> y_pred = knn.predict(X_test);

        // Display results
        std::cout << "Predictions:\n";
        for (size_t i = 0; i < y_pred.size(); ++i)
        {
            std::cout << "Sample " << i
                      << " -> Predicted: " << y_pred[i]
                      << ", Expected: " << y_expected[i]
                      << " [IQ: " << X_test[i][0]
                      << ", CGPA: " << X_test[i][1] << "]\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
    }

    return 0;
}
