#include <stdio.h>
#include <stdlib.h>
#include "mnist_loader.h"
#include "neural_net.h"

#define IMAGE_INDEX 0
#define TRAINING_SIZE 50000
#define TEST_SIZE 10000
#define EPOCHS 30
#define MINI_BATCH_SIZE 32
#define LEARNING_RATE 0.1

double** one_hot_encode_all(const int* labels, size_t n) {
    double** encoded = malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        encoded[i] = malloc(10 * sizeof(double));
        for (int j = 0; j < 10; j++) {
            encoded[i][j] = (labels[i] == j) ? 1.0 : 0.0;
        }
    }
    return encoded;
}

void free_one_hot_labels(double** labels, size_t n) {
    for (size_t i = 0; i < n; i++) {
        free(labels[i]);
    }
    free(labels);
}

void print_image(const double* image, int rows, int cols) {
    const char shades[] = " .:-=+*#%@"; // 10 levels of intensity
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pixel = image[i * cols + j];  // 0 to 1
            int index = (int)(pixel * 9); // Scale to 0-9
            printf("%c", shades[index]);
        }
        printf("\n");
    }
}

int argmax(const double* output, int n) {
    int max_i = 0;
    double max_val = output[0];
    for (int i = 1; i < n; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_i = i;
        }
    }
    return max_i;
}

int main() {
    const char* train_image_file = "data/train-images.idx3-ubyte";
    const char* train_label_file = "data/train-labels.idx1-ubyte";

    const char* test_image_file = "data/t10k-images.idx3-ubyte";
    const char* test_label_file = "data/t10k-labels.idx1-ubyte";

    int rows, cols;
    double** train_images = load_mnist_images(train_image_file, TRAINING_SIZE, &rows, &cols);
    int* train_labels = load_mnist_labels(train_label_file, TRAINING_SIZE);

    if (!train_images || !train_labels) {
        fprintf(stderr, "Failed to load MNIST dataset.\n");
        return 1;
    }

    double** test_images = load_mnist_images(test_image_file, TEST_SIZE, &rows, &cols);
    int* test_labels = load_mnist_labels(test_label_file, TEST_SIZE);

    if (!test_images || !test_labels) {
        fprintf(stderr, "Failed to load MNIST test dataset.\n");
        return 1;
    }

    // Convert labels to one-hot vectors
    double** one_hot_train_labels = one_hot_encode_all(train_labels, TRAINING_SIZE);

    int input_size = rows * cols;
    int layer_sizes[] = { input_size, 30, 10 };

    NeuralNet* nn = nn_create(3, layer_sizes);
    if (!nn) {
        fprintf(stderr, "Failed to create neural network.\n");
        return 1;
    }

    printf("Neural network created with %d layers.\n", nn->num_layers);
    printf("Evaluating on test set before training...\n");

    int correct = nn_evaluate(nn, (const double**) test_images, test_labels, TEST_SIZE);
    printf("Accuracy on test set: %d / %d (%.2f%%)\n", correct, TEST_SIZE, 100.0 * correct / TEST_SIZE);

    printf("Starting training...\n");
    nn_train(nn, (const double**) train_images, (const double**) one_hot_train_labels,
             TRAINING_SIZE, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE);

    printf("\nTraining complete!\n");

    printf("Evaluating on test set after training...\n");
    correct = nn_evaluate(nn, (const double**) test_images, test_labels, TEST_SIZE);
    printf("Accuracy on test set: %d / %d (%.2f%%)\n", correct, TEST_SIZE, 100.0 * correct / TEST_SIZE);

    // Test a single image
    int test_index = 0;
    printf("\nImage (label = %d):\n\n", test_labels[test_index]);
    print_image(test_images[test_index], rows, cols);

    double output[10];
    nn_feedforward(nn, test_images[test_index], output);

    int predicted = argmax(output, 10);
    printf("\nPredicted label: %d\n", predicted);

    // Cleanup
    nn_free(nn);

    free_mnist_images(train_images, TRAINING_SIZE);
    free_one_hot_labels(one_hot_train_labels, TRAINING_SIZE);
    free(train_labels);

    free_mnist_images(test_images, TEST_SIZE);
    free(test_labels);

    return 0;
}