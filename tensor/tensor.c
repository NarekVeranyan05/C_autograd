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

    int size = 1;
    for (int i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    t->data = malloc(sizeof(double) * size);

    t->shape = malloc(sizeof(int) * ndims);
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

const double epsilon = 0.0001;

struct Tensor *tensor_invert(struct Tensor *t)
{
    struct Tensor *parents[1] = {t};
    struct Tensor *inverse = create_tensor(t->shape, t->ndims, parents, 1, INVERSE);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
        inverse->data[i] = 1 / (t->data[i] + epsilon);

    return inverse;
}

struct Tensor *tensor_negate(struct Tensor *t)
{
    const double epsilon = 0.0001;

    struct Tensor *parents[1] = {t};
    struct Tensor *negation = create_tensor(t->shape, t->ndims, parents, 1, NEGATION);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
        negation->data[i] = -(t->data[i]);

    return negation;
}

struct Tensor *tensor_log(struct Tensor *t)
{
    struct Tensor *parents[1] = {t};
    struct Tensor *logged = create_tensor(t->shape, t->ndims, parents, 1, INVERSE);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
        logged->data[i] = log(t->data[i]);

    return logged;
}

struct Tensor *tensor_exp(struct Tensor *t)
{
    struct Tensor *parents[1] = {t};
    struct Tensor *exponential = create_tensor(t->shape, t->ndims, parents, 1, INVERSE);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
        exponential->data[i] = exp(t->data[i]);

    return exponential;
}

struct Tensor *tensor_ReLU(struct Tensor *t)
{
    struct Tensor *parents[1] = {t};
    struct Tensor *tensor_activated = create_tensor(t->shape, t->ndims, parents, 1, RELU);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
        tensor_activated->data[i] = (t->data[i] <= 0) ? 0 : t->data[i];

    return tensor_activated;
}

struct Tensor *tensor_sigmoid(struct Tensor *t)
{
    struct Tensor *parents[1] = {t};
    struct Tensor *tensor_activated = create_tensor(t->shape, t->ndims, parents, 1, SIGMOID);

    int size = 1;
    for (int i = 0; i < t->ndims; i++)
        size *= t->shape[i];

    for (int i = 0; i < size; i++)
    {
        tensor_activated->data[i] = 1 / ( 1 + exp( -t->data[i] ) );
    }

    return tensor_activated;
}

struct Tensor *tensor_softmax(struct Tensor *t)
{
    assert(t->ndims == 2 && (t->shape[0] == 1 || t->shape[1] == 1));

    int size = t->shape[0] * t->shape[1];

    // computing the exponents for scalars
    struct Tensor *parents[1] = {t};
    struct Tensor *exponents = create_tensor(t->shape, t->ndims, parents, 1, SOFTMAX);
    double sum_of_exponents = 0;
    for (int i = 0; i < size; i++)
    {
        exponents->data[i] = exp(t->data[i]);
        sum_of_exponents += exponents->data[i];
    }
    for (int i = 0; i < size; i++)
        exponents->data[i] /= sum_of_exponents;

    return exponents;
}

struct Tensor *tensor_selector(struct Tensor *t, int index)
{
    assert(t->ndims == 2 && (t->shape[0] == 1 || t->shape[1] == 1));

    struct Tensor *selected = NULL;

    if (t->shape[0] != 1) // n x 1
    {
        // initialising indicator
        int shape[2] = { 1, t->shape[0] }; // 1 x n
        struct Tensor *indicator = create_tensor(shape, t->ndims, NULL, 0, UNDEF);
        init_empty(indicator);
        indicator->data[index] = 1;

        selected = tensor_matmul(indicator, t);
    }
    else // 1 x n
    {
        // initialising indicator
        int shape[2] = { t->shape[1], 1 }; // n x 1
        struct Tensor *indicator = create_tensor(shape, t->ndims, NULL, 0, UNDEF);
        init_empty(indicator);
        indicator->data[index] = 1;

        selected = tensor_matmul(t, indicator);
    }

    return selected;
}

bool check_equal(struct Tensor *a, struct Tensor *b)
{
    const double epsilon = 1e-4f;

    assert(a->ndims == b->ndims);
    for (int i = 0; i < a->ndims; i++)
        assert(a->shape[i] == b->shape[i]);

    bool equal = true;
    int size = 1;
    for (int i = 0; i < a->ndims; i++)
        size *= a->shape[i];

    for (int i = 0; i < size; i++)
        equal &= ( fabs(a->data[i] - b->data[i]) < epsilon );

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
