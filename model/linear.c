//
// Created by Narek Veranian on 2026-08-03.
//

#include "linear.h"

#include <stdlib.h>
#include <time.h>

#include "tensor.h"

typedef struct Linear {
    struct Tensor *W;
    struct Tensor *b;
    struct Tensor *(*activation)(struct Tensor *t);
} Linear;

// Desired range boundaries for the random floats
static const double max_val =  1e-6;
static const double min_val = -1e-6;

Linear *create_linear_layer(struct Tensor *(*activation)(struct Tensor *t), int input_size, int num_neurons)
{
    srand((unsigned int)time(NULL));

    Linear *ll = malloc(sizeof(Linear));

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

    ll->activation = activation;

    return ll;
}

struct Tensor *forward(const Linear *lin_layer, struct Tensor *input) {
    struct Tensor *pre_activation = tensor_add(tensor_matmul(input, lin_layer->W), lin_layer->b);
    return lin_layer->activation(pre_activation);
}

