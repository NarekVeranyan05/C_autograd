//
// Created by Narek Veranian on 2026-07-20.
//
#include "./backward.h"
#include "./tensor.h"
#include <stdlib.h>
#include <string.h>

void backward_sum(struct Tensor *self)
{
    // contributing gradient for parent 1
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);
    struct Tensor *grad1 = self->parents[0]->grad;

    struct Tensor *sum1 = tensor_add(grad1, self->grad);
    tensor_destruct(grad1);
    self->parents[0]->grad = sum1;

    // contributing gradient for parent 2
    if (self->parents[1]->grad == NULL)
        init_grad(self->parents[1]);
    struct Tensor *grad2 = self->parents[1]->grad;

    struct Tensor *sum2 = tensor_add(grad2, self->grad);
    tensor_destruct(grad2);
    self->parents[1]->grad = sum2;
}

void backward_matmul(struct Tensor *self)
{
    // contributing gradient for parent 1
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    // constructing gradient
    struct Tensor *parent2_transposed = tensor_transpose(self->parents[1]);
    struct Tensor *grad1 = tensor_matmul(self->grad, parent2_transposed);

    // adding gradient
    struct Tensor *sum1 = tensor_add(grad1, self->parents[0]->grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum1;


    // contributing gradient for parent 2
    if (self->parents[1]->grad == NULL)
        init_grad(self->parents[1]);

    // constructing gradient
    struct Tensor *parent1_transposed = tensor_transpose(self->parents[0]);
    struct Tensor *grad2 = tensor_matmul(parent1_transposed, self->grad);

    // adding gradient
    struct Tensor *sum2 = tensor_add(grad2, self->parents[1]->grad);
    tensor_destruct(self->parents[1]->grad);
    self->parents[1]->grad = sum2;
}

void backward_transpose(struct Tensor *self)
{
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *self_grad_transposed = tensor_transpose(self->grad);
    struct Tensor *sum = tensor_add(self_grad_transposed, self->parents[0]->grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum;
}