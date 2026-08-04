//
// Created by Narek Veranian on 2026-08-03.
//

#include <stddef.h>

#include "model.h"

#include <assert.h>
#include <stdlib.h>

#include "loss.h"
#include "layers.h"
#include "tensor.h"
#include "../assets/queue.h"

typedef struct Model {
    struct Queue *queue;
    enum Loss loss_type;
} Model;

Model *create_model(enum Loss loss_type)
{
    Model *model = malloc(sizeof(Model));
    model->queue = create_queue();
    model->loss_type = loss_type;

    return model;
}

void destruct_model(Model *model)
{
    assert(model != NULL);

    destruct_queue(model->queue);
    free(model);
}

static void destruct_linear_layer_voided(void *ll_voided)
{
    Linear_layer *ll = ll_voided;
    destruct_linear_layer(ll);
}

void add_linear_layer(const Model *model, const int n_neurons, const enum Activation activation)
{
    const Linear_layer *last_layer = peek_tail(model->queue);
    const int last_out_size = num_neurons(last_layer);

    enqueue(model->queue, create_linear_layer(
        activation, last_out_size, n_neurons
    ), destruct_linear_layer_voided);
}

struct Tensor *model_forward(const Model *model, struct Tensor *input)
{
    const struct Queue *temp = create_queue();

    const Linear_layer *first_layer = peek_head(model->queue);
    struct Tensor *output = forward(first_layer, input);
    enqueue(temp, dequeue(model->queue), NULL);

    while (!isEmpty(model->queue))
    {
        const Linear_layer *curr_layer = peek_head(model->queue);
        output = forward(curr_layer, output);
        enqueue(temp, dequeue(model->queue), NULL);
    }

    return output;
}

struct Tensor *compute_loss(const Model *model, struct Tensor *input, const int true_output)
{
    struct Tensor *output = model_forward(model, input);

    switch (model->loss_type) {
        case MULTI_CLASS_CROSS_ENTROPY:
            return multi_class_cross_entropy(output, true_output);
        case BINARY_CROSS_ENTROPY:
            return binary_cross_entropy(output, true_output);
    }
}