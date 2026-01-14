#include "Tensor.h"

int main() {
    Tensor entrada = Tensor::random({1000, 20, 20});
    entrada.view({1000, 400});

    Tensor W1 = Tensor::random({400, 100});
    Tensor b1 = Tensor::ones({1000, 100});
    Tensor capa1 = matmul(entrada, W1) + b1;

    ReLU relu;
    capa1 = capa1.apply(relu);

    Tensor W2 = Tensor::random({100, 10});
    Tensor b2 = Tensor::ones({1000, 10});
    Tensor salida = matmul(capa1, W2) + b2;

    Sigmoid sigmoid;
    salida = salida.apply(sigmoid);

    cout << "Dimensiones finales: " << salida.getShape()[0] << "x" << salida.getShape()[1] << endl;

    return 0;
}