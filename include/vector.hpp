// vector.hpp

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <vector>
#include <cmath>

// A custom Vector class that wraps std::vector<double> and provides basic vector operations
class Vector
{
public:
    std::vector<double> data; // Internal representation of the vector

    // === Constructors ===

    Vector() : data() {} // Default constructor

    Vector(size_t size) : data(size, 0.0) {} // Initialize vector of given size with all zeros

    Vector(const std::vector<double> &values) : data(values) {} // Construct from std::vector<double>

    Vector(const Vector &other) : data(other.data) {} // Copy constructor

    // Assignment operator
    Vector &operator=(const Vector &other)
    {
        if (this != &other)
        {
            data = other.data;
        }
        return *this;
    }

    // === Element Access ===

    // Non-const index access with bounds checking
    double &operator[](size_t index)
    {
        if (index >= data.size())
        {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // Const index access with bounds checking
    const double &operator[](size_t index) const
    {
        if (index >= data.size())
        {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // === Comparisons ===

    // Check for equality with another Vector (uses std::vector's operator==)
    bool operator==(const Vector &other) const
    {
        return data == other.data;
    }

    // === Arithmetic Operations ===

    // Element-wise addition
    Vector operator+(const Vector &other) const
    {
        if (data.size() != other.data.size())
        {
            throw std::invalid_argument("Vectors must be of same size");
        }
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Element-wise subtraction
    Vector operator-(const Vector &other) const
    {
        if (data.size() != other.data.size())
        {
            throw std::invalid_argument("Vectors must be of same size");
        }
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Scalar division
    Vector operator/(int scalar) const
    {
        if (scalar == 0)
        {
            throw std::invalid_argument("Division by zero");
        }
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    // === Vector Operations ===

    // Dot product with another vector
    double dot(const Vector &other) const
    {
        if (data.size() != other.data.size())
        {
            throw std::invalid_argument("Vectors must be of same size");
        }
        double result = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            result += data[i] * other.data[i];
        }
        return result;
    }

    // L2 norm (magnitude of vector)
    double norm() const
    {
        double sum = 0.0;
        for (const auto &value : data)
        {
            sum += value * value;
        }
        return std::sqrt(sum);
    }

    // Euclidean distance between this and another vector
    double euclideanDistance(const Vector &other) const
    {
        if (data.size() != other.data.size())
        {
            throw std::invalid_argument("Vectors must be of same size");
        }
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            double diff = data[i] - other.data[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Squared Euclidean distance (no sqrt for efficiency)
    double euclideanSquaredDistance(const Vector &other) const
    {
        if (data.size() != other.data.size())
        {
            throw std::invalid_argument("Vectors must be of same size");
        }
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            double diff = data[i] - other.data[i];
            sum += diff * diff;
        }
        return sum;
    }

    // === Utility Functions ===

    size_t size() const
    {
        return data.size();
    }

    void resize(size_t newSize)
    {
        data.resize(newSize, 0.0);
    }

    void push_back(double value)
    {
        data.push_back(value);
    }

    void pop_back()
    {
        if (data.empty())
            throw std::out_of_range("Cannot pop from an empty vector");
        data.pop_back();
    }
};

#endif // VECTOR_HPP
