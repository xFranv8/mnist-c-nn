#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "neural_net.h"

static double rand_normal() {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

NeuralNet* nn_create(int num_layers, int* sizes) {
    NeuralNet* nn = malloc(sizeof(NeuralNet));
    if (!nn) return NULL;

    nn->num_layers = num_layers;
    nn->sizes = malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        nn->sizes[i] = sizes[i];
    }

    // Allocate biases (no biases for input layer)
    nn->biases = malloc((num_layers - 1) * sizeof(double*));
    for (int i = 1; i < num_layers; i++) {
        nn->biases[i - 1] = malloc(sizes[i] * sizeof(double));
        for (int j = 0; j < sizes[i]; j++) {
            nn->biases[i - 1][j] = rand_normal();
        }
    }

    // Allocate weights
    nn->weights = malloc((num_layers - 1) * sizeof(double**));
    for (int i = 1; i < num_layers; i++) {
        int rows = sizes[i];
        int cols = sizes[i - 1];
        nn->weights[i - 1] = malloc(rows * sizeof(double*));
        for (int j = 0; j < rows; j++) {
            nn->weights[i - 1][j] = malloc(cols * sizeof(double));
            for (int k = 0; k < cols; k++) {
                nn->weights[i - 1][j][k] = rand_normal();
            }
        }
    }

    return nn;
}

void nn_free(NeuralNet* nn) {
    if (!nn) return;

    for (int i = 1; i < nn->num_layers; i++) {
        free(nn->biases[i - 1]);
    }
    free(nn->biases);

    for (int i = 1; i < nn->num_layers; i++) {
        int rows = nn->sizes[i];
        for (int j = 0; j < rows; j++) {
            free(nn->weights[i - 1][j]);
        }
        free(nn->weights[i - 1]);
    }
    free(nn->weights);

    free(nn->sizes);
    free(nn);
}

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

static double sigmoid_prime(double z) {
    double s = 1.0 / (1.0 + exp(-z));
    return s * (1 - s);
}

static void shuffle(double** a, double** b, size_t size) {
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        double* tmp_a = a[i]; a[i] = a[j]; a[j] = tmp_a;
        double* tmp_b = b[i]; b[i] = b[j]; b[j] = tmp_b;
    }
}

void nn_feedforward(NeuralNet* nn, const double* input, double* output) {
    const double* a = input;

    // We'll reuse a temporary buffer for intermediate activations
    double* current_output = malloc(nn->sizes[nn->num_layers - 1] * sizeof(double));

    for (int l = 1; l < nn->num_layers; l++) {
        int rows = nn->sizes[l];
        int cols = nn->sizes[l - 1];

        double* z = malloc(rows * sizeof(double));
        for (int i = 0; i < rows; i++) {
            z[i] = nn->biases[l - 1][i];
            for (int j = 0; j < cols; j++) {
                z[i] += nn->weights[l - 1][i][j] * a[j];
            }
            z[i] = sigmoid(z[i]);
        }

        a = z;

        if (l == nn->num_layers - 1) {
            for (int i = 0; i < rows; i++) {
                output[i] = z[i];
            }
        }

        if (l != 1) free((void*)a);  // Free previous z
    }

    free(current_output);
}

void nn_train(NeuralNet* nn, const double** training_inputs, const double** training_labels,
    size_t training_size, int epochs, int mini_batch_size, double eta) {
    srand(time(NULL));  // For shuffling

    int num_layers = nn->num_layers;

    // Allocate temporary gradients
    double*** nabla_w = malloc((num_layers - 1) * sizeof(double**));
    double** nabla_b = malloc((num_layers - 1) * sizeof(double*));
    for (int l = 1; l < num_layers; l++) {
    int rows = nn->sizes[l];
    int cols = nn->sizes[l - 1];
    nabla_w[l - 1] = matrix_alloc(rows, cols);
    nabla_b[l - 1] = malloc(rows * sizeof(double));
    }

    // Create shuffled copies of training sets
    double** inputs = (double**) training_inputs;
    double** labels = (double**) training_labels;

    for (int epoch = 0; epoch < epochs; epoch++) {
    shuffle(inputs, labels, training_size);

    for (size_t start = 0; start < training_size; start += mini_batch_size) {
    size_t end = (start + mini_batch_size < training_size) ? (start + mini_batch_size) : training_size;
    size_t batch_size = end - start;

    // Reset gradients
    for (int l = 1; l < num_layers; l++) {
        int rows = nn->sizes[l];
        int cols = nn->sizes[l - 1];
        vector_zero(nabla_b[l - 1], rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                nabla_w[l - 1][i][j] = 0.0;
            }
        }
    }

    // Accumulate gradients
    for (size_t i = start; i < end; i++) {
        // FORWARD PASS
        double** activations = malloc(num_layers * sizeof(double*));
        double** zs = malloc((num_layers - 1) * sizeof(double*));
        activations[0] = malloc(nn->sizes[0] * sizeof(double));
        vector_copy(inputs[i], activations[0], nn->sizes[0]);

        for (int l = 1; l < num_layers; l++) {
            int size = nn->sizes[l];
            int prev = nn->sizes[l - 1];

            zs[l - 1] = malloc(size * sizeof(double));
            activations[l] = malloc(size * sizeof(double));

            matrix_vector_mul(nn->weights[l - 1], activations[l - 1], zs[l - 1], size, prev);
            vector_add(zs[l - 1], nn->biases[l - 1], zs[l - 1], size);  // z = w·a + b
            for (int j = 0; j < size; j++) {
                activations[l][j] = sigmoid(zs[l - 1][j]);
            }
        }

        // BACKWARD PASS
        double* delta = malloc(nn->sizes[num_layers - 1] * sizeof(double));
        int out_size = nn->sizes[num_layers - 1];
        for (int j = 0; j < out_size; j++) {
            double a = activations[num_layers - 1][j];
            delta[j] = (a - training_labels[i][j]) * sigmoid_prime(zs[num_layers - 2][j]);
        }

        vector_add(nabla_b[num_layers - 2], delta, nabla_b[num_layers - 2], out_size);
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < nn->sizes[num_layers - 2]; k++) {
                nabla_w[num_layers - 2][j][k] += delta[j] * activations[num_layers - 2][k];
            }
        }

        // Backpropagate to previous layers
        for (int l = num_layers - 2; l > 0; l--) {
            int size = nn->sizes[l];
            int next = nn->sizes[l + 1];

            double* sp = malloc(size * sizeof(double));
            for (int j = 0; j < size; j++) {
                sp[j] = sigmoid_prime(zs[l - 1][j]);
            }

            double* new_delta = malloc(size * sizeof(double));
            for (int j = 0; j < size; j++) {
                new_delta[j] = 0.0;
                for (int k = 0; k < next; k++) {
                    new_delta[j] += nn->weights[l][k][j] * delta[k];
                }
                new_delta[j] *= sp[j];
                nabla_b[l - 1][j] += new_delta[j];
                for (int k = 0; k < nn->sizes[l - 1]; k++) {
                    nabla_w[l - 1][j][k] += new_delta[j] * activations[l - 1][k];
                }
            }

            free(delta);
            delta = new_delta;
            free(sp);
        }

        free(delta);
        for (int l = 0; l < num_layers; l++) free(activations[l]);
        for (int l = 0; l < num_layers - 1; l++) free(zs[l]);
        free(activations);
        free(zs);
    }

    // Apply gradients
    for (int l = 1; l < num_layers; l++) {
        int rows = nn->sizes[l];
        int cols = nn->sizes[l - 1];
        for (int i = 0; i < rows; i++) {
            nn->biases[l - 1][i] -= (eta / batch_size) * nabla_b[l - 1][i];
            for (int j = 0; j < cols; j++) {
                nn->weights[l - 1][i][j] -= (eta / batch_size) * nabla_w[l - 1][i][j];
            }
        }
    }
    }

    printf("Epoch %d complete\n", epoch + 1);
    }

    // Cleanup
    for (int l = 1; l < num_layers; l++) {
    free(nabla_b[l - 1]);
    matrix_free(nabla_w[l - 1], nn->sizes[l]);
    }
    free(nabla_b);
    free(nabla_w);
}

int nn_evaluate(NeuralNet* nn, const double** test_inputs, const int* test_labels, size_t test_size) {
    int correct = 0;
    int output_size = nn->sizes[nn->num_layers - 1];
    double* output = malloc(output_size * sizeof(double));
    if (!output) return 0;

    for (size_t i = 0; i < test_size; i++) {
        nn_feedforward(nn, test_inputs[i], output);

        // Buscar el índice de la mayor probabilidad (predicción)
        int predicted = 0;
        double max_val = output[0];
        for (int j = 1; j < output_size; j++) {
            if (output[j] > max_val) {
                max_val = output[j];
                predicted = j;
            }
        }

        if (predicted == test_labels[i]) {
            correct++;
        }
    }

    free(output);
    return correct;
}