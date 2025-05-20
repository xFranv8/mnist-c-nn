#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stddef.h>

/**
 * Loads MNIST images from a given IDX file.
 * @param filename Path to the IDX image file
 * @param num_images Number of images to load (set to 0 to load all)
 * @param rows Output: number of rows per image
 * @param cols Output: number of columns per image
 * @return Pointer to an array of image pointers, each image is a double array [rows * cols]
 */
double** load_mnist_images(const char* filename, size_t num_images, int* rows, int* cols);

/**
 * Loads MNIST labels from a given IDX label file.
 * @param filename Path to the IDX label file
 * @param num_labels Number of labels to load (set to 0 to load all)
 * @return Pointer to an array of integer labels
 */
int* load_mnist_labels(const char* filename, size_t num_labels);

/**
 * Frees the memory allocated for the MNIST image array.
 * @param images Pointer to the image array
 * @param num_images Number of images in the array
 */
void free_mnist_images(double** images, size_t num_images);

/**
 * Converts a label into a one-hot encoded vector (e.g., 3 → [0 0 0 1 0 0 0 0 0 0]).
 * @param label Integer label (0–9)
 * @param out Output array (must be size 10)
 */
void one_hot_encode(int label, double* out);

#endif // MNIST_LOADER_H