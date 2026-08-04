//
// Created by Narek Veranian on 2026-08-03.
//

#ifndef UNTITLED_LINEAR_H
#define UNTITLED_LINEAR_H

#include "tensor.h"

typedef struct Linear_layer Linear_layer;
typedef struct Predictor_layer Predictor_layer;

enum Activation {
    RELU,
    SIGMOID,
    SOFTMAX
};
#define ACTIVATION_COUNT 3

/**
 * Creates a linear layer.
 * @param activation the activation function the layer must pass data through
 * @param input_size the number of columns in the input matrix
 * @param num_neurons the number of neurons in the layer
 * @return a linear layer, as configured, with weights initialised randomly
 */
Linear_layer *create_linear_layer(enum Activation activation, int input_size, int num_neurons);

/**
 * Destructs a linear layer.
 * @param ll the linear layer to destruct
 */
void destruct_linear_layer(Linear_layer *ll);

/**
 * Computes the output of the linear layer
 * @param lin_layer the linear layer to pass input through
 * @param input the input data
 * @return
 */
struct Tensor *forward(const Linear_layer *lin_layer, struct Tensor *input);

int get_input_size(const Linear_layer *lin_layer);
int num_neurons(const Linear_layer *lin_layer);

#endif //UNTITLED_LINEAR_H