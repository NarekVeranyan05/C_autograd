//
// Created by Narek Veranian on 2026-08-03.
//

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "optimiser.h"
#include "tensor.h"
#include "../model/model.h"
#include "topological_sort.h"
#include "../assets/linked_list.h"

/**
 * a Stochastic Gradient Descent optimiser.
 * the params array has three entries:
 * params[0] = the learning rate (positive)
 * params[1] = the forgetting rate
 * params[2] = the number of epochs
 */
typedef struct SGD {
    struct Optimiser *optimiser;
} SGD;

SGD *create_sgd(const Model *model, struct Tensor **data, int *true_outputs, int sample_size, double learning_rate, double forgetting_rate, int num_epochs)
{
    assert(model != NULL);
    assert(data != NULL);
    assert(sample_size >= 1);
    assert(learning_rate > 0);
    assert(num_epochs >= 1);

    SGD *sgd = malloc(sizeof(SGD));

    sgd->optimiser->params = malloc(sizeof(double) * 2);
    sgd->optimiser->params[0] = learning_rate;
    sgd->optimiser->params[1] = forgetting_rate;
    sgd->optimiser->params[2] = num_epochs;

    sgd->optimiser->model = model;
    sgd->optimiser->data = data;
    sgd->optimiser->true_outputs = true_outputs;
    sgd->optimiser->data_size = sample_size;

    return sgd;
}

void sgd_optimise(const SGD *self)
{
    assert(self != NULL);

    double metric_estimate = 0;
    bool stabilised = false;

    for (int epoch = 0; epoch < self->optimiser->params[1] && !stabilised; epoch++)
    {
        for (int i = 0; i < self->optimiser->data_size && !stabilised; i++)
        {
            struct Tensor *curr_object = self->optimiser->data[i];
            int true_output = self->optimiser->true_outputs[i];
            struct Tensor *loss = compute_loss(self->optimiser->model, curr_object, true_output);

            // gradient step
            sgd_make_step(self, loss);

            // estimate metric
            if (i == 0) metric_estimate = loss->data[0];
            else {
                const double old_metric_estimate = metric_estimate;

                metric_estimate =
                    (1 - self->optimiser->params[1]) * metric_estimate +
                    self->optimiser->params[1] * loss->data[0];

                stabilised = (fabs(old_metric_estimate - metric_estimate) < 0.001);
            }

            tensor_destruct(loss, true);
        }
    }
}

static void sgd_make_step(const SGD *sgd, struct Tensor *loss)
{
    const struct Linked_list *sorted = topological_sort(loss);

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


