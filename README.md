# Tarea 1: Proyecto Tensor++

**Nombre:** Sebastian Vir Briceño Gora  
**Código:** 202310002

## Descripción
Este proyecto implementa una librería de Tensores en C++ para operaciones matemáticas y procesamiento de datos. El objetivo es gestionar grandes volúmenes de información de manera eficiente usando memoria dinámica y semántica de movimiento.

## Conceptos de Programación III Aplicados
* **Gestión de Memoria (RAII):** Uso de punteros `double*` con reservación en el Heap y liberación automática en el destructor para evitar fugas de memoria.
* **Semántica de Movimiento:** Implementación de constructores de movimiento y operadores de asignación por movimiento (`operator=`) para optimizar el rendimiento al transferir datos.
* **Polimorfismo:** Uso de una clase base abstracta `TensorTransform` para aplicar funciones de activación (ReLU y Sigmoid) de forma genérica.
* **Sobrecarga de Operadores:** Implementación de operadores matemáticos para facilitar la suma de tensores y multiplicaciones.

## Ejecución del Caso Real
El archivo `main.cpp` ejecuta el flujo de trabajo solicitado:
1. Crea un tensor aleatorio de 1000x20x20.
2. Cambia su forma a 1000x400 mediante el método `view`.
3. Aplica capas de pesos, bias y funciones de activación.
4. Muestra en consola las dimensiones finales: **1000x10**.