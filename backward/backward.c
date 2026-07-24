//
// Created by Narek Veranian on 2026-07-20.
//
#include "./backward.h"

#include <assert.h>

#include "./tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

void backward_mul(struct Tensor *self)
{
    // contributing gradient to parent 1
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *grad1 = tensor_mul(self->grad, self->parents[1]);
    struct Tensor *sum1 = tensor_add(grad1, self->parents[0]->grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum1;

    // contributing gradient to parent 2
    if (self->parents[1]->grad == NULL)
        init_grad(self->parents[1]);

    struct Tensor *grad2 = tensor_mul(self->grad, self->parents[0]);
    struct Tensor *sum2 = tensor_add(grad2, self->parents[1]->grad);
    tensor_destruct(self->parents[1]->grad);
    self->parents[1]->grad = sum2;
}

void backward_scale(struct Tensor *self)
{
    // contributing gradient to the scaling factor (parent 1)
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *mul = tensor_mul(self->grad, self->parents[1]);
    int shape[2] = {1, 1};
    struct Tensor *grad1 = create_tensor(shape, 2, NULL, 0, UNDEF);
    init_empty(grad1);

    int size = 1;
    for (int i = 0; i < self->parents[1]->ndims; i++)
        size *= self->parents[1]->shape[i];
    for (int i = 0; i < size; i++)
        grad1->data[0] += mul->data[i];

    struct Tensor *sum1 = tensor_add(grad1, self->parents[0]->grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum1;

    // contributing gradient to the scaled tensor (parent 2)
    if (self->parents[1]->grad == NULL)
        init_grad(self->parents[1]);

    struct Tensor *grad2 = tensor_scale(self->parents[0]->data[0], self->grad);
    struct Tensor *sum2 = tensor_add(grad2, self->parents[1]->grad);
    tensor_destruct(self->parents[1]->grad);
    self->parents[1]->grad = sum2;
}

void backward_sigmoid(struct Tensor *self)
{
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    int size = 1;
    for (int i = 0; i < self->grad->ndims; i++)
        size  *= self->grad->shape[i];

    // creating unit tensor (all ones)
    struct Tensor *parents[1] = {self};
    struct Tensor *ones = create_tensor(self->shape, self->ndims, parents, 1, UNDEF);
    for (int i = 0; i < size; i++)
        ones->data[i] = 1;

    // exponential^(-margin)
    struct Tensor *exp = tensor_exp(tensor_scale(-1, self->parents[0]));
    struct Tensor *denom = tensor_add(ones, exp);

    struct Tensor *grad = tensor_mul(
        self->grad,
        tensor_mul( exp, tensor_invert( tensor_mul(denom, denom)) )
    );

    // adding the gradient
    struct Tensor *sum = tensor_add(self->parents[0]->grad, grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum;
}

void backward_ReLU(struct Tensor *self)
{
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    int size = 1;
    for (int i = 0; i < self->grad->ndims; i++)
        size  *= self->grad->shape[i];

    for (int i = 0; i < size; i++)
        self->parents[0]->grad->data[i] += self->grad->data[i] * (self->parents[0]->data[i] > 0);
}

// NOTE: we assume that softmax was applied to a vector, and only 1 element was selected
// this is the backprop through that single selected element (parent)
void backward_softmax(struct Tensor *self)
{
    assert(self->ndims == 2 && (self->shape[0] == 1 || self->shape[1] == 1));

    int size = self->shape[0] * self->shape[1];

    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *grad = NULL;

    int shape[2] = {size, size};
    struct Tensor *parents[2] = {self->parents[0], self};
    struct Tensor *softmax_logit_grad = create_tensor(shape, 2, parents, 2, UNDEF);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                softmax_logit_grad->data[i * size + j] = self->data[i] * (1 - self->data[i]);
            else
                softmax_logit_grad->data[i * size + j] = -self->data[i] * self->data[j];
        }
    }

    if (self->shape[0] != 1) // n x 1
    {
        grad = tensor_transpose( tensor_matmul(
            tensor_transpose(self->grad),
            softmax_logit_grad
        ) );
    }
    else
    {
        grad = tensor_matmul(self->grad, softmax_logit_grad);
    }

    // adding the gradient
    struct Tensor *sum = tensor_add(self->parents[0]->grad, grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum;
}


void backward_log(struct Tensor *self)
{
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *grad = tensor_mul(self->grad, tensor_invert(self->parents[0]));
    struct Tensor *sum = tensor_add(self->parents[0]->grad, grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum;
}

void backward_exp(struct Tensor *self)
{
    if (self->parents[0]->grad == NULL)
        init_grad(self->parents[0]);

    struct Tensor *grad = tensor_mul(self->grad, self);
    struct Tensor *sum = tensor_add(grad, self->parents[0]->grad);
    tensor_destruct(self->parents[0]->grad);
    self->parents[0]->grad = sum;
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