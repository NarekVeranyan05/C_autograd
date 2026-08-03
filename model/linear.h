//
// Created by Narek Veranian on 2026-08-03.
//

#ifndef UNTITLED_LINEAR_H
#define UNTITLED_LINEAR_H

#include "tensor.h"

typedef struct Linear Linear;

/**
 * Creates a linear layer.
 * @param activation the activation function the layer must pass data through
 * @param input_size the number of columns in the input matrix
 * @param num_neurons the number of neurons in the layer
 * @return a linear layer, as configured, with weights initialised randomly
 */
Linear *create_linear_layer(struct Tensor *(*activation)(struct Tensor *t), int input_size, int num_neurons);

/**
 * Computes the output of the linear layer
 * @param lin_layer the linear layer to pass input through
 * @param input the input data
 * @return
 */
struct Tensor *forward(const Linear *lin_layer, struct Tensor *input);

#endif //UNTITLED_LINEAR_H