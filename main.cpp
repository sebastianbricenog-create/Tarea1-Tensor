#include <iostream>
#include "Tensor.h"

int main() {
    try {
        Tensor t1 = Tensor::arange(0, 10, 2);
        t1.print_shape();

        Tensor t2 = t1.view({5, 1});
        t2.print_shape();

        Tensor v1({3}, {1, 0, -1});
        Tensor v2({3}, {2, 5, 2});
        std::cout << "Dot: " << v1.dot(v2) << std::endl;

        Tensor v3 = v2 - v1;

        Tensor error = v1 * t2;

    } catch (const std::exception& e) {
        std::cerr << "Excepcion: " << e.what() << std::endl;
    }

    return 0;
}