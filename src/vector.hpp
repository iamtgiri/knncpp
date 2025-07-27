// vector.hpp

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <iostream>
#include <vector>
#include <cmath>

class Vector
{
public:
    std::vector<double> data;

    Vector() : data() {}
    Vector(size_t size) : data(size, 0.0) {}
    Vector(const std::vector<double> &values) : data(values) {}
    Vector(const Vector &other) : data(other.data) {}
    Vector &operator=(const Vector &other)
    {
        if (this != &other)
        {
            data = other.data;
        }
        return *this;
    }
    double &operator[](size_t index)
    {
        if (index >= data.size())
        {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }
    bool operator==(const Vector &other) const
    {
        return data == other.data; // relies on std::vector's operator==
    }

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

    const double &operator[](size_t index) const
    {
        if (index >= data.size() || index < 0)
        {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }
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
    double norm() const
    {
        double sum = 0.0;
        for (const auto &value : data)
        {
            sum += value * value;
        }
        return std::sqrt(sum);
    }

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
    };
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
};
#endif // VECTOR_HPP