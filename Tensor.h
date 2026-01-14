#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

class TensorTransform {
public:
    virtual double apply(double x) const = 0;
};

class Tensor {
private:
    double* ptr;
    vector<size_t> shape;
    size_t total_size;

public:
    Tensor(vector<size_t> s);
    ~Tensor();
    Tensor(const Tensor& otro);
    Tensor(Tensor&& otro) noexcept;

    // Agregamos esto para que no salga el error de la foto
    Tensor& operator=(Tensor&& otro) noexcept;

    void view(vector<size_t> nuevas_dimensiones);

    Tensor operator+(const Tensor& otro);
    Tensor operator*(double escalar);
    Tensor apply(const TensorTransform& t);

    static Tensor zeros(vector<size_t> s);
    static Tensor ones(vector<size_t> s);
    static Tensor random(vector<size_t> s);

    size_t getSize() const { return total_size; }
    vector<size_t> getShape() const { return shape; }

    friend Tensor matmul(const Tensor& a, const Tensor& b);
};

class ReLU : public TensorTransform {
    double apply(double x) const override { return x > 0 ? x : 0; }
};

class Sigmoid : public TensorTransform {
    double apply(double x) const override { return 1.0 / (1.0 + exp(-x)); }
};

#endif