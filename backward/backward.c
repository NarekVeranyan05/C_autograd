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
