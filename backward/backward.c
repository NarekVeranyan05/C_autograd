//
// Created by Narek Veranian on 2026-07-20.
//
#include "./backward.h"
#include <stdlib.h>

void backward_sum(struct Tensor *self)
{
    self->parents[0]->grad = create_tensor(
        self->parents[0]->shape,
        self->parents[0]->ndims,
        NULL,
        0,
        GRAD
    );

    self->parents[1]->grad = create_tensor(
        self->parents[1]->shape,
        self->parents[1]->ndims,
        NULL,
        0,
        GRAD
    );

}