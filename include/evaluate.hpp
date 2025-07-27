// evaluate.hpp

#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <vector>
#include <stdexcept>

// Computes the accuracy of predictions compared to true labels
// Accuracy = (Number of Correct Predictions) / (Total Predictions)
//
// Parameters:
// - y_test: Ground truth labels
// - y_pred: Predicted labels by the model
//
// Returns:
// - A double in range [0, 1], representing accuracy as a proportion
//
// Throws:
// - std::invalid_argument if input vectors are of unequal length
double accuracy_score(const std::vector<int> &y_test, const std::vector<int> &y_pred)
{
    // Sanity check: both vectors must be of same size
    if (y_test.size() != y_pred.size())
    {
        throw std::invalid_argument("y_test and y_pred must have the same length.");
    }

    int correct = 0; // Counter for correct predictions

    // Compare each predicted label with the corresponding true label
    for (int i = 0; i < y_pred.size(); ++i)
    {
        if (y_pred[i] == y_test[i])
        {
            ++correct;
        }
    }

    // Return accuracy as a floating-point ratio
    return static_cast<double>(correct) / y_test.size();
}

#endif // EVALUATE_HPP
