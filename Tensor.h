#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <algorithm>

class Tensor {
private:
    double* data;
    std::vector<int> shape;
    int total_size;

public:
    Tensor();
    Tensor(std::vector<int> s);
    Tensor(std::vector<int> s, std::initializer_list<double> values);

    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    static Tensor arange(double start, double end, double step = 1.0);
    static Tensor concat(const Tensor& a, const Tensor& b, int axis);
    Tensor unsqueeze(int axis) const;
    double dot(const Tensor& other) const;
    Tensor view(std::vector<int> new_shape) const;

    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    void print_shape() const;
    int get_total_size() const { return total_size; }
};

#endif