# MNIST Neural Network in C

This project implements a simple feedforward neural network in C from scratch to classify handwritten digits from the MNIST dataset. No external ML libraries or frameworks are used.

## Features

- Full implementation of forward and backpropagation
- RELU activation function for hidden layers
- Softmax activation function for the output layer
- Cross-entropy loss function
- Trains using mini-batch stochastic gradient descent (SGD)
- Loads IDX-formatted MNIST dataset directly
- Written in pure ANSI C for portability and performance

## Build & Run

```bash
make ./mnist_nn
./mnist_nn
```
