//
// Created by Narek Veranian on 2026-07-20.
//
// Each backward function receives the tensor for the current node in the computational graph.
// Then, according to the rule that generated the tensor, it will compute the gradients
// of the parents for the node.


#ifndef UNTITLED_BACKWARD_H
#define UNTITLED_BACKWARD_H

#include "../tensor/tensor.h"

/**
 * Computes gradients for parent-addends
 * @param self – the current tensor generated as a sum
 */
void backward_sum(struct Tensor *self);

/**
 * Computes gradients for parent-multiplicands (matmul)
 * @param self – the current tensor generated as a product
 */
void backward_matmul(struct Tensor *self);

/**
 * Computes gradients for parent through tanh
 * @param self – the current tensor generated as a hyperbolic tangent from the parent
 */
void backward_tanh(struct Tensor *self);

/**
 * Computes gradients for parent through sigmoid
 * @param self – the current tensor generated as a sigmoid from the parent
 */
void backward_sigmoid(struct Tensor *self);

/**
 * Computes gradients for parent through ReLU
 * @param self – the current tensor generated as a ReLU from the parent
 */
void backward_ReLU(struct Tensor *self);

/**
 * Computes gradients for parent through transpose
 * @param self – the current tensor generated as a ReLU from the parent
 */
void backward_transpose(struct Tensor *self);

#endif //UNTITLED_BACKWARD_H
