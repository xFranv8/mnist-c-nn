#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/**
 * Allocates a matrix of size (rows x cols).
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the allocated matrix
 */
double** matrix_alloc(int rows, int cols);

/**
 * Frees a matrix previously allocated with matrix_alloc.
 * @param mat Pointer to the matrix
 * @param rows Number of rows in the matrix
 */
void matrix_free(double** mat, int rows);

/**
 * Multiplies a matrix A (rows x cols) by a vector v (cols x 1).
 * @param A Matrix A
 * @param v Vector to multiply
 * @param out Output vector (must be pre-allocated, size = rows)
 * @param rows Number of rows in A
 * @param cols Number of columns in A (and size of v)
 */
void matrix_vector_mul(double** A, const double* v, double* out, int rows, int cols);

/**
 * Adds two vectors element-wise: out = a + b
 * @param a First input vector
 * @param b Second input vector
 * @param out Output vector (must be pre-allocated)
 * @param n Length of the vectors
 */
void vector_add(const double* a, const double* b, double* out, int n);

/**
 * Subtracts two vectors element-wise: out = a - b
 * @param a First input vector
 * @param b Second input vector
 * @param out Output vector (must be pre-allocated)
 * @param n Length of the vectors
 */
void vector_sub(const double* a, const double* b, double* out, int n);

/**
 * Element-wise multiplication of two vectors: out = a * b
 * @param a First input vector
 * @param b Second input vector
 * @param out Output vector (must be pre-allocated)
 * @param n Length of the vectors
 */
void vector_mul(const double* a, const double* b, double* out, int n);

/**
 * Multiplies a vector by a scalar: out = scalar * a
 * @param a Input vector
 * @param scalar Scalar value
 * @param out Output vector (must be pre-allocated)
 * @param n Length of the vector
 */
void vector_scalar_mul(const double* a, double scalar, double* out, int n);

/**
 * Applies a function element-wise to a vector: out[i] = func(a[i])
 * (Modifies the input vector in-place)
 * @param a Input/output vector
 * @param n Length of the vector
 * @param func Function pointer (e.g., sigmoid)
 */
void vector_apply(double* a, int n, double (*func)(double));

/**
 * Copies vector a into vector b.
 * @param a Source vector
 * @param b Destination vector (must be pre-allocated)
 * @param n Length of the vectors
 */
void vector_copy(const double* a, double* b, int n);

/**
 * Computes the dot product of two vectors.
 * @param a First input vector
 * @param b Second input vector
 * @param n Length of the vectors
 * @return Dot product result
 */
double vector_dot(const double* a, const double* b, int n);

/**
 * Initializes a vector to all zeros.
 * @param v Vector to initialize
 * @param n Length of the vector
 */
void vector_zero(double* v, int n);

#endif // MATRIX_H