//
// Created by Narek Veranian on 2026-08-03.
//
#pragma once

#include "tensor.h"

#ifndef UNTITLED_OPTIMISER_H
#define UNTITLED_OPTIMISER_H

enum Optimisation_type {
    MAX,
    MIN
};

/**
 * An Optimiser performs minimisation / maximisation of
 * the target function.
 */
struct Optimiser {
    enum Optimisation_type type;
    double *params;
    struct Tensor *target;
    struct Tensor **data;
    int data_size;
};

typedef struct SGD SGD;
typedef struct Adam Adam;
typedef struct Momentum Momentum;

/**
 * Creates a Stochastic Gradient Descent optimiser.
 * @param target the target function to minimise / maximise
 * @param data the data on which to optimise
 * @param data_size the number of data elements (tensors)
 * @param learning_rate the rate by which to move weights toward anti-gradient
 * @param num_epochs the number of times to pass over the entire dataset
 * @return a Stochastic Gradient Descent optimiser
 */
SGD *create_sgd(
    struct Tensor *target,
    struct Tensor **data,
    int data_size,
    double learning_rate,
    int num_epochs
);

/**
 * Creates a Stochastic Gradient Descent optimiser.
 * @param target the target function to minimise / maximise
 * @param data the data on which to optimise
 * @param data_size the number of data elements (tensors)
 * @param gamma the smoothing factor for a moving average over momentum
 * @param alpha the smoothing factor for a moving average over squared gradients
 * @param learning_rate the rate by which to move weights toward anti-gradient
 * @param num_epochs the number of times to pass over the entire dataset
 * @return a Stochastic Gradient Descent optimiser
 */
Adam *create_adam(
    struct Tensor *target,
    struct Tensor **data,
    int data_size,
    double gamma,
    double alpha,
    double learning_rate,
    int num_epochs
);

/**
 * Creates a Stochastic Gradient Descent optimiser.
 * @param gamma the smoothing factor for a moving average over momentum
 * @param target the target function to minimise / maximise
 * @param data the data on which to optimise
 * @param data_size the number of data elements (tensors)
 * @return a Stochastic Gradient Descent optimiser
 */
Momentum *create_momentum(double learning_rate, struct Tensor *target, struct Tensor **data, int data_size);

#endif //UNTITLED_OPTIMISER_H
