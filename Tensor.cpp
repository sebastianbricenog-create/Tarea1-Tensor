#include "Tensor.h"

Tensor::Tensor(vector<size_t> s) : shape(s) {
    total_size = 1;
    for (size_t i = 0; i < s.size(); i++) total_size *= s[i];
    ptr = new double[total_size];
    for (size_t i = 0; i < total_size; i++) ptr[i] = 0.0;
}

Tensor::~Tensor() {
    if (ptr != nullptr) delete[] ptr;
}

Tensor::Tensor(const Tensor& otro) {
    shape = otro.shape;
    total_size = otro.total_size;
    ptr = new double[total_size];
    for (size_t i = 0; i < total_size; i++) ptr[i] = otro.ptr[i];
}

Tensor::Tensor(Tensor&& otro) noexcept : ptr(otro.ptr), shape(otro.shape), total_size(otro.total_size) {
    otro.ptr = nullptr;
    otro.total_size = 0;
}

// Esta funcion resuelve el error "deleted function" de tu captura
Tensor& Tensor::operator=(Tensor&& otro) noexcept {
    if (this != &otro) {
        delete[] ptr;
        ptr = otro.ptr;
        shape = otro.shape;
        total_size = otro.total_size;
        otro.ptr = nullptr;
        otro.total_size = 0;
    }
    return *this;
}

void Tensor::view(vector<size_t> nuevas_dimensiones) {
    size_t nuevo_total = 1;
    for (size_t i = 0; i < nuevas_dimensiones.size(); i++) nuevo_total *= nuevas_dimensiones[i];
    if (nuevo_total == total_size) shape = nuevas_dimensiones;
}

Tensor Tensor::operator+(const Tensor& otro) {
    Tensor res(shape);
    for (size_t i = 0; i < total_size; i++) res.ptr[i] = ptr[i] + otro.ptr[i];
    return move(res);
}

Tensor Tensor::operator*(double escalar) {
    Tensor res(shape);
    for (size_t i = 0; i < total_size; i++) res.ptr[i] = ptr[i] * escalar;
    return move(res);
}

Tensor Tensor::apply(const TensorTransform& t) {
    Tensor res(shape);
    for (size_t i = 0; i < total_size; i++) res.ptr[i] = t.apply(ptr[i]);
    return move(res);
}

Tensor Tensor::zeros(vector<size_t> s) { return Tensor(s); }

Tensor Tensor::ones(vector<size_t> s) {
    Tensor t(s);
    for (size_t i = 0; i < t.total_size; i++) t.ptr[i] = 1.0;
    return move(t);
}

Tensor Tensor::random(vector<size_t> s) {
    Tensor t(s);
    for (size_t i = 0; i < t.total_size; i++) t.ptr[i] = (double)rand() / RAND_MAX;
    return move(t);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    size_t f = a.shape[0];
    size_t ca = a.shape[1];
    size_t cb = b.shape[1];
    Tensor res({f, cb});
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < cb; j++) {
            double s = 0;
            for (size_t k = 0; k < ca; k++) s += a.ptr[i * ca + k] * b.ptr[k * cb + j];
            res.ptr[i * cb + j] = s;
        }
    }
    return move(res);
}