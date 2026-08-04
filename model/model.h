//
// Created by Narek Veranian on 2026-08-03.
//

#ifndef UNTITLED_MODEL_H
#define UNTITLED_MODEL_H

#include "./layers.h"
#include "./loss.h"

typedef struct Model Model;

/**
 * Creates a new model.
 * @param loss_type The loss function type
 * @return the model structure
 */
Model *create_model(enum Loss loss_type);

void destruct_model(Model *model);

/**
 * Adds a new linear layer to the model.
 * @param model the model to append a linear layer to
 * @param n_neurons the number of neurons in the layer
 * @param activation the activation function of the layer
 */
void add_linear_layer(const Model *model, int n_neurons, enum Activation activation);

/**
 * Passes an input through the model.
 * @param model the model to compute the output
 * @param input the input to pass through the model
 * @return the output of the model
 */
struct Tensor *model_forward(const Model *model, struct Tensor *input);

struct Tensor *compute_loss(const Model *model, struct Tensor *input, const int true_output);

#endif //UNTITLED_MODEL_H
