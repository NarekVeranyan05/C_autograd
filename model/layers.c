//
// Created by Narek Veranian on 2026-08-03.
//

#include "layers.h"

#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"

struct Tensor *(*activations[ACTIVATION_COUNT])(struct Tensor *) = {
    [RELU]    = tensor_ReLU,
    [SIGMOID] = tensor_sigmoid,
    [SOFTMAX] = tensor_softmax
};

typedef struct Linear_layer {
    struct Tensor *W;
    struct Tensor *b;
    struct Tensor *(*activation)(struct Tensor *t);
} Linear_layer;

// Desired range boundaries for the random floats
static const double max_val =  1e-6;
static const double min_val = -1e-6;

Linear_layer *create_linear_layer(enum Activation activation, int input_size, int num_neurons)
{
    srand((unsigned int)time(NULL));

    Linear_layer *ll = malloc(sizeof(Linear_layer));

    // (1) NEURON WEIGHTS
    int shape1[2] = {input_size, num_neurons};
    ll->W = create_tensor(shape1, 2, NULL, 0, OP_UNDEF);

    for (int i = 0; i < input_size * num_neurons; i++)
    {
        // generate a double between 0 and 1
        double scale = (double)rand() / (double) RAND_MAX;

        // scale and shift to [min_val, max_val] range
        ll->W->data[i] = min_val + scale * (max_val - min_val);
    }

    // (2) NEURON BIAS
    int shape2[2] = {1, num_neurons};
    ll->b = create_tensor(shape2, 2, NULL, 0, OP_UNDEF);

    for (int i = 0; i < num_neurons; i++)
    {
        // generate a double between 0 and 1
        double scale = (double)rand() / (double) RAND_MAX;

        // scale and shift to [min_val, max_val] range
        ll->b->data[i] = min_val + scale * (max_val - min_val);
    }

    ll->activation = activations[activation];

    return ll;
}

void destruct_linear_layer(Linear_layer *ll)
{
    assert(ll != NULL);

    tensor_destruct(ll->W, true);
    tensor_destruct(ll->b, true);
    free(ll);
}

struct Tensor *forward(const Linear_layer *lin_layer, struct Tensor *input) {
    struct Tensor *pre_activation = tensor_add(tensor_matmul(input, lin_layer->W), lin_layer->b);
    return lin_layer->activation(pre_activation);
}

int get_input_size(const Linear_layer *lin_layer)
{
    return lin_layer->W->shape[0];
}

int num_neurons(const Linear_layer *lin_layer)
{
    return lin_layer->W->shape[1];
}

