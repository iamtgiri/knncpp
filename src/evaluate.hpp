// evaluate.hpp


#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include <vector>
#include <stdexcept>

double accuracy_score(const std::vector<int> &y_test, const std::vector<int> &y_pred)
{
    if(y_test.size() != y_pred.size())
    {
        throw std::invalid_argument("y_test and y_pred must have the same length.");
    }
    int correct = 0;
    for(int i =0; i<y_pred.size(); ++i)
    {
        if(y_pred[i] == y_test[i])
        {
            ++correct;
        }
    }
    return static_cast<double>(correct) / y_test.size();
}

#endif // EVALUATE_HPP