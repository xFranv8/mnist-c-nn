#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

/**
 * Structure representing a simple feedforward neural network.
 */
typedef struct {
    int num_layers;        // Total number of layers in the network
    int* sizes;            // Array with the number of neurons in each layer
    double** biases;       // Bias vectors for each layer (excluding input layer)
    double*** weights;     // Weight matrices between layers
} NeuralNet;

/**
 * Creates a new neural network with the specified layer sizes.
 * @param num_layers Number of layers in the network
 * @param sizes Array of layer sizes (e.g., [784, 30, 10])
 * @return Pointer to the created NeuralNet structure
 */
NeuralNet* nn_create(int num_layers, int* sizes);

/**
 * Frees all memory allocated for the given neural network.
 * @param nn Pointer to the NeuralNet to free
 */
void nn_free(NeuralNet* nn);

/**
 * Performs feedforward computation on the given input.
 * @param nn Pointer to the NeuralNet
 * @param input Array of input values (size: sizes[0])
 * @param output Pre-allocated array to store the output (size: sizes[num_layers - 1])
 */
void nn_feedforward(NeuralNet* nn, const double* input, double* output);

/**
 * Trains the neural network using stochastic gradient descent.
 * @param nn Pointer to the NeuralNet
 * @param training_inputs Array of pointers to training input arrays
 * @param training_labels Array of pointers to expected output arrays
 * @param training_size Number of training samples
 * @param epochs Number of training epochs
 * @param mini_batch_size Size of each mini-batch
 * @param eta Learning rate
 */
void nn_train(NeuralNet* nn, const double** training_inputs, const double** training_labels,
              size_t training_size, int epochs, int mini_batch_size, double eta);

/**
 * Evaluates the network accuracy on the test set.
 * @param nn Pointer to the NeuralNet
 * @param test_inputs Array of pointers to test input arrays
 * @param test_labels Array of expected labels (as integers)
 * @param test_size Number of test samples
 * @return Number of correct predictions
 */
int nn_evaluate(NeuralNet* nn, const double** test_inputs, const int* test_labels, size_t test_size);

#endif // NEURAL_NET_H