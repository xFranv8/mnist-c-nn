#include <stdlib.h>
#include <string.h>
#include "matrix.h"

double** matrix_alloc(int rows, int cols) {
    double** mat = malloc(rows * sizeof(double*));
    if (!mat) return NULL;

    for (int i = 0; i < rows; i++) {
        mat[i] = malloc(cols * sizeof(double));
        if (!mat[i]) {
            // Free previously allocated rows on failure
            for (int j = 0; j < i; j++) {
                free(mat[j]);
            }
            free(mat);
            return NULL;
        }
    }

    return mat;
}

void matrix_free(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void matrix_vector_mul(double** A, const double* v, double* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            out[i] += A[i][j] * v[j];
        }
    }
}

void vector_add(const double* a, const double* b, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vector_sub(const double* a, const double* b, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void vector_mul(const double* a, const double* b, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void vector_scalar_mul(const double* a, double scalar, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = scalar * a[i];
    }
}

void vector_apply(double* a, int n, double (*func)(double)) {
    for (int i = 0; i < n; i++) {
        a[i] = func(a[i]);
    }
}

void vector_copy(const double* a, double* b, int n) {
    memcpy(b, a, n * sizeof(double));
}

double vector_dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_zero(double* v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 0.0;
    }
}