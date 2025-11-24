// include\evaluate.hpp

#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <vector>
#include <stdexcept>


/**
 * Calculates the accuracy score between true labels and predicted labels.
 * @param y_test Vector of true labels.
 * @param y_pred Vector of predicted labels.
 * @return Accuracy as a double value between 0 and 1.
 * @throws std::invalid_argument if the sizes of y_test and y_pred do not match.
 */
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
