//
// Created by Narek Veranian on 2026-07-19.
//

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "./tensor.h"
#include <math.h>

#include <stdio.h>

struct Tensor *create_tensor(int *shape, int ndims, struct Tensor **parents, int num_parents, enum Operation op)
{
    assert(ndims >= 2);

    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->data = malloc(sizeof(double) * shape[0] * shape[1]);
    t->shape = malloc(sizeof(double) * ndims);
    for (int i = 0; i < ndims; i++)
    {
        t->shape[i] = shape[i];
    }
    t->ndims = ndims;

    t->parents = malloc(sizeof(struct Tensor *) * num_parents);
    for (int i = 0; i < num_parents; i++)
    {
        t->parents[i] = parents[i];
    }
    t->num_parents = num_parents;

    t->op = op;

    t->grad = NULL;
    t->requires_grad = false;

    return t;
}

void init_empty(struct Tensor *t)
{
    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
    {
        t->data[i] = 0;
    }
}

void init_grad(struct Tensor *t)
{
    t->grad = create_tensor(t->shape, t->ndims, &t, 1, GRAD);
    init_empty(t->grad);
}

struct Tensor *tensor_matmul(struct Tensor *a, struct Tensor *b)
{
    assert(a->ndims == b->ndims && a->ndims == 2 && a->shape[1] == b->shape[0]);
    int shape[2] = {a->shape[0], b->shape[1]};

    struct Tensor *parents[2] = {a, b};

    struct Tensor *prod = create_tensor(shape, a->ndims, parents, 2, SUM);
    for (int i = 0; i < a->shape[0]; i++) // i-th row of the product
    {
        for (int j = 0; j < b->shape[1]; j++) // j-th column of the product
        {
            // prod(i, j) equals to the dot product of the i-th row and j-th column of the tensors a and b
            float sum = 0;
            for (int h = 0; h < a->shape[1]; h++)
            {
                sum += a->data[i * a->shape[1] + h] * b->data[h * b->shape[1] + j];
            }
            prod->data[i * prod->shape[1] + j] = sum;
        }
    }

    return prod;
}

struct Tensor *tensor_mul(struct Tensor *a, struct Tensor *b)
{
    assert(a->ndims == b->ndims);
    for (int i = 0; i < a->ndims; i++)
    {
        assert(a->shape[i] == b->shape[i]);
    }

    struct Tensor *parents[2] = {a, b};

    struct Tensor *t = create_tensor(a->shape, a->ndims, parents, 2, MUL);
    int size = 1;
    for (int i = 0; i < a->ndims; i++)
        size *= a->shape[i];

    for (int i = 0; i < size; i++)
        t->data[i] = a->data[i] * b->data[i];

    return t;
}

struct Tensor *tensor_add(struct Tensor *a, struct Tensor *b)
{
    int size = 1;
    assert(a->ndims == b->ndims);
    for (int i = 0; i < a->ndims; i++)
    {
        size *= a->shape[i];
        assert(a->shape[i] == b->shape[i]);
    }

    struct Tensor *parents[2] = {a, b};

    struct Tensor *t = create_tensor(a->shape, a->ndims, parents, 2, SUM);
    for (int i = 0; i < size; i++)
    {
        t->data[i] = a->data[i] + b->data[i];
    }

    return t;
}

struct Tensor *tensor_transpose(struct Tensor *t)
{
    assert(t->ndims == 2);

    int new_shape[2] = {t->shape[1], t->shape[0]};

    struct Tensor *parents[1] = {t};

    struct Tensor *transposed = create_tensor(new_shape, t->ndims, parents, 1, TRANSPOSE);

    for (int i = 0; i < t->shape[0]; i++)
    {
        for (int j = 0; j < t->shape[1]; j++)
        {
            transposed->data[j * transposed->shape[1] + i] = t->data[i * t->shape[1] + j];
        }
    }

    return transposed;
}

bool check_equal(struct Tensor *a, struct Tensor *b)
{
    assert(a->ndims == b->ndims);
    for (int i = 0; i < a->ndims; i++)
        assert(a->shape[i] == b->shape[i]);

    bool equal = true;
    int size = 1;
    for (int i = 0; i < a->ndims; i++)
        size *= a->shape[i];

    for (int i = 0; i < size; i++)
        equal &= ( fabs(a->data[i] - b->data[i]) < 0.0001 );

    return equal;
}

void tensor_destruct(struct Tensor *t)
{
    if (t->grad != NULL)
    {
        free(t->grad->data);
        free(t->grad->shape);
        free(t->grad->parents);
        free(t->grad);
    }

    free(t->data);
    free(t->shape);
    free(t->parents);
    free(t);
}