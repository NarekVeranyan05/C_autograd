//
// Created by Narek Veranian on 2026-08-03.
//

#include <assert.h>
#include <c++/v1/cstdlib>

#include "optimiser.h"
#include "tensor.h"
#include "topological_sort.h"
#include "../stack/linked_list.h"

/**
 * a Stochastic Gradient Descent optimiser.
 * the params array has two entries:
 * params[0] = the learning rate (positive)
 * params[1] = the number of epochs
 */
typedef struct SGD {
    struct Optimiser *optimiser;
} SGD;

SGD *create_sgd(struct Tensor *target, struct Tensor **data, int data_size, double learning_rate, int num_epochs)
{
    assert(target != NULL);
    assert(data != NULL);
    assert(data_size >= 1);
    assert(learning_rate > 0);
    assert(num_epochs >= 1);

    SGD *sgd = malloc(sizeof(SGD));

    sgd->optimiser->params = malloc(sizeof(double) * 2);
    sgd->optimiser->params[0] = learning_rate;
    sgd->optimiser->params[1] = num_epochs;

    sgd->optimiser->target = target;
    sgd->optimiser->data = data;
    sgd->optimiser->data_size = data_size;

    return sgd;
}

void sgd_optimise(const SGD *self)
{
    assert(self != NULL);

    for (int epoch = 0; epoch < self->optimiser->params[1]; epoch++)
    {
        for (int i = 0; i < self->optimiser->data_size; i++)
        {

        }
    }
}

static void sgd_make_step(const SGD *sgd)
{
    struct Linked_list *sorted = topological_sort(sgd->optimiser->target);

    for (int i = 0; i < size(sorted); i++)
    {
        struct Tensor *curr_tensor = get_at(sorted, i);
        if (curr_tensor->requires_grad)
        {
            const double learning_rate = (sgd->optimiser->type == MAX) ?
                sgd->optimiser->params[0] :
                -sgd->optimiser->params[0];

            struct Tensor *grad_step = tensor_scale(learning_rate, curr_tensor->grad);
            struct Tensor *new_tensor = tensor_add(curr_tensor, grad_step);

            // freeing previous gradient
            tensor_destruct(curr_tensor->grad, true);
            curr_tensor->grad = NULL;

            // updating data
            free(curr_tensor->data);
            curr_tensor->data = new_tensor->data;

            // freeing
            tensor_destruct(new_tensor, false);
            tensor_destruct(grad_step, true);
        }
    }
}


