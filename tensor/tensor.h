//
// Created by Narek Veranian on 2026-07-20.
//

#ifndef UNTITLED_TENSOR_H
#define UNTITLED_TENSOR_H

#include <stdbool.h>

enum Operation {
    SUM,
    MATMUL,
    MUL,
    TRANSPOSE,
    SIGMOID,
    RELU,
    EXP,
    LOG,
    POW2,
    GRAD
};

struct Tensor {
    double *data;
    int *shape;
    int ndims;

    struct Tensor **parents;
    int num_parents;
    void (*backward)(struct Tensor *self);
    enum Operation op;

    struct Tensor *grad;
    bool requires_grad;
};

/**
 * Creates a new tensor out of parent tensors
 * @param shape - the size of the tensor along all its axes
 * @param ndims - the number of axes of the tensor. Must be greater than or equal to 2
 * @param parents - the parent tensors that generated current tensor
 * @param num_parents - the amount of parent tensors
 * @param op - the operations performed on parent tensors to get current tensor
 * @return - the child tensor
 */
struct Tensor *create_tensor(int *shape, int ndims, struct Tensor *parents[2], int num_parents, enum Operation op);


/**
 * Sets all entries of a tensor to zero.
 * @param t - the tensor to set to empty
 */
void init_empty(struct Tensor *t);

/**
 * Initialises empty gradient for the tensor.
 * @param t - the tensor to initialise the gradient for
 */
void init_grad(struct Tensor *t);

/**
 * Performs matrix multiplication.
 * @param a - the first multiplicand-matrix
 * @param b - the second multiplicand-matrix
 * @return - a @ b
 * @note a and b must have ndims = 2 and
 * the number of columns of a must equal the number of rows of b.
 */
struct Tensor *tensor_matmul(struct Tensor *a, struct Tensor *b);

/**
 * Performs Hadamard product of tensors/
 * @param a - the first multiplicand-tensor
 * @param b - the second multiplicand-tensor
 * @return - a * b
 * @note a and b must have same ndims and same sizes along each axis.
 */
struct Tensor *tensor_mul(struct Tensor *a, struct Tensor *b);

/**
 * Adds two tensors together.
 * @param a - the first addend-tensor
 * @param b - the second addend-tensor
 * @return - a + b
 * @note a and b must have same ndims and same sizes along each axis.
 */
struct Tensor *tensor_add(struct Tensor *a, struct Tensor *b);

/**
 * Transposes a matrix.
 * @param t - the matrix to transpose
 * @return - t.transpose
 * @note t must have ndims less than or equal to 2.
 */
struct Tensor *tensor_transpose(struct Tensor *t);

/**
 * Checks if two tensors equal each other
 * @param a - the first tensor
 * @param b - the second tensor
 * @return true if a->data = b->data, false otherwise
 * @note a and b must have same ndims and same sizes along each axis.
 */
bool check_equal(struct Tensor *a, struct Tensor *b);

/**
 * Destructs a tensor
 * @param t - the tensor to destruct
 */
void tensor_destruct(struct Tensor *t);

#endif //UNTITLED_TENSOR_H
