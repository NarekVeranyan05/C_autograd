//
// Created by Narek Veranian on 2026-08-03.
//

#ifndef UNTITLED_LOSS_H
#define UNTITLED_LOSS_H

#include "../tensor/tensor.h"

enum Loss {
    MULTI_CLASS_CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
};

/**
 * Computes the negative log-likelihood over multi-class output.
 * @param output the output of the model (class probabilities)
 * @param true_class_index the index of the true class of the object
 * @return the negative log-likelihood of the output
 */
struct Tensor *multi_class_cross_entropy(struct Tensor *output, int true_class_index);

/**
 * Computes the negative log-likelihood over classification.
 * @param positive_class_probability the probability of the positive class
 * @param true_class the true class of the object
 * @return
 */
struct Tensor *binary_cross_entropy(struct Tensor *positive_class_probability, int true_class);

#endif //UNTITLED_LOSS_H
