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

struct Tensor *create_tensor(int *shape, int ndims, struct Tensor *parents[2], int num_parents, enum Operation op);

struct Tensor *tensor_matmul(struct Tensor *a, struct Tensor *b);

struct Tensor *tensor_mul(struct Tensor *a, struct Tensor *b);

struct Tensor *tensor_transpose(struct Tensor *t);

bool check_equal(struct Tensor *a, struct Tensor *b);

#endif //UNTITLED_TENSOR_H
