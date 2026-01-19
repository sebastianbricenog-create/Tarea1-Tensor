#include "Tensor.h"
#include <cmath>

Tensor::Tensor() : data(nullptr), total_size(0) {}

Tensor::Tensor(std::vector<int> s) : shape(s) {
    total_size = 1;
    for (int d : shape) total_size *= d;
    data = new double[total_size]();
}

Tensor::Tensor(std::vector<int> s, std::initializer_list<double> values) : Tensor(s) {
    if (values.size() != (size_t)total_size)
        throw std::invalid_argument("Error de dimensiones");
    std::copy(values.begin(), values.end(), data);
}

Tensor::~Tensor() {
    delete[] data;
}

Tensor::Tensor(const Tensor& other) : shape(other.shape), total_size(other.total_size) {
    data = new double[total_size];
    std::copy(other.data, other.data + total_size, data);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        shape = other.shape;
        total_size = other.total_size;
        data = new double[total_size];
        std::copy(other.data, other.data + total_size, data);
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data), shape(std::move(other.shape)), total_size(other.total_size) {
    other.data = nullptr;
    other.total_size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data = other.data;
        shape = std::move(other.shape);
        total_size = other.total_size;
        other.data = nullptr;
        other.total_size = 0;
    }
    return *this;
}

Tensor Tensor::arange(double start, double end, double step) {
    if (step <= 0) throw std::invalid_argument("Step invalido");
    int n = static_cast<int>(std::ceil((end - start) / step));
    Tensor t({n});
    for (int i = 0; i < n; ++i) t.data[i] = start + i * step;
    return t;
}

Tensor Tensor::view(std::vector<int> new_shape) const {
    int new_total = 1;
    for (int d : new_shape) new_total *= d;
    if (new_total != total_size)
        throw std::runtime_error("Dimensiones incompatibles");
    Tensor t(*this);
    t.shape = new_shape;
    return t;
}

Tensor Tensor::unsqueeze(int axis) const {
    if (axis < 0 || axis > (int)shape.size()) throw std::out_of_range("Eje invalido");
    std::vector<int> new_shape = shape;
    new_shape.insert(new_shape.begin() + axis, 1);
    return view(new_shape);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape != other.shape) throw std::invalid_argument("Error en resta");
    Tensor res(shape);
    for (int i = 0; i < total_size; ++i) res.data[i] = data[i] - other.data[i];
    return res;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape != other.shape) throw std::invalid_argument("Error en multiplicacion");
    Tensor res(shape);
    for (int i = 0; i < total_size; ++i) res.data[i] = data[i] * other.data[i];
    return res;
}

double Tensor::dot(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Error en dot");
    double res = 0;
    for (int i = 0; i < total_size; ++i) res += data[i] * other.data[i];
    return res;
}

void Tensor::print_shape() const {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i)
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : "x");
    std::cout << "]" << std::endl;
}

Tensor Tensor::concat(const Tensor& a, const Tensor& b, int axis) {
    if (axis != 0) throw std::runtime_error("Eje no soportado");
    std::vector<int> new_shape = a.shape;
    new_shape[0] = a.shape[0] + b.shape[0];
    Tensor res(new_shape);
    std::copy(a.data, a.data + a.total_size, res.data);
    std::copy(b.data, b.data + b.total_size, res.data + a.total_size);
    return res;
}