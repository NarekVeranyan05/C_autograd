//
// Created by Narek Veranian on 2026-08-03.
//

#include <c++/v1/cstdlib>

#include "optimiser.h"

typedef struct Adam {
    struct Optimiser optimiser;
} Adam;

Adam *create_adam(double gamma, double alpha, double learning_rate, struct Tensor *target, struct Tensor **data, int data_size)
{
    Adam *adam = malloc(sizeof(Adam));

    adam->optimiser.params = malloc(sizeof(double) * 3);
    adam->optimiser.params[0] = gamma;
    adam->optimiser.params[1] = alpha;
    adam->optimiser.params[2] = learning_rate;

    adam->optimiser.target = target;
    adam->optimiser.data = data;
    adam->optimiser.data_size = data_size;

    return adam;
}