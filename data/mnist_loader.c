#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "mnist_loader.h"

static uint32_t read_uint32(FILE* f) {
    uint8_t bytes[4];
    fread(bytes, sizeof(uint8_t), 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

double** load_mnist_images(const char* filename, size_t num_images, int* rows, int* cols) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Unable to open image file %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32(f);
    if (magic != 2051) {
        fprintf(stderr, "Invalid magic number in image file: %u\n", magic);
        fclose(f);
        return NULL;
    }

    uint32_t total_images = read_uint32(f);
    uint32_t image_rows = read_uint32(f);
    uint32_t image_cols = read_uint32(f);

    if (num_images == 0 || num_images > total_images) {
        num_images = total_images;
    }

    double** images = malloc(num_images * sizeof(double*));
    size_t image_size = image_rows * image_cols;

    for (size_t i = 0; i < num_images; i++) {
        images[i] = malloc(image_size * sizeof(double));
        for (size_t j = 0; j < image_size; j++) {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, f);
            images[i][j] = pixel / 255.0;  // normalize to [0, 1]
        }
    }

    *rows = image_rows;
    *cols = image_cols;

    fclose(f);
    return images;
}

int* load_mnist_labels(const char* filename, size_t num_labels) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Unable to open label file %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32(f);
    if (magic != 2049) {
        fprintf(stderr, "Invalid magic number in label file: %u\n", magic);
        fclose(f);
        return NULL;
    }

    uint32_t total_labels = read_uint32(f);
    if (num_labels == 0 || num_labels > total_labels) {
        num_labels = total_labels;
    }

    int* labels = malloc(num_labels * sizeof(int));
    for (size_t i = 0; i < num_labels; i++) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, f);
        labels[i] = (int)label;
    }

    fclose(f);
    return labels;
}

void free_mnist_images(double** images, size_t num_images) {
    for (size_t i = 0; i < num_images; i++) {
        free(images[i]);
    }
    free(images);
}

void one_hot_encode(int label, double* out) {
    for (int i = 0; i < 10; i++) {
        out[i] = (i == label) ? 1.0 : 0.0;
    }
}