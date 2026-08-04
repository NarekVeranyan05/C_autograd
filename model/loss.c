//
// Created by Narek Veranian on 2026-08-03.
//

#include "loss.h"

#include <assert.h>
#include <stddef.h>

struct Tensor *cross_entropy(struct Tensor *output, const int true_class_index)
{
    assert(output != NULL);
    assert(output->ndims == 2 &&
        (output->shape[0] == 1 || output->shape[1] == 1));

    return tensor_scale(-1, tensor_log(tensor_selector(output, true_class_index)));
}

struct Tensor *binary_cross_entropy(struct Tensor *positive_class_probability, int true_class)
{
    assert(positive_class_probability != NULL);
    assert(positive_class_probability->ndims == 2 &&
        positive_class_probability->shape[0] == 1 && positive_class_probability->shape[1] == 1);
    assert(true_class == 0 || true_class == 1);

    int shape[2] = {1, 1};
    struct Tensor *ones = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    struct Tensor *negative_class_probability = tensor_add(
        ones,
        tensor_scale(-1, positive_class_probability)
    );

    return tensor_add(
         tensor_scale(true_class, tensor_log(positive_class_probability)),
        tensor_scale(1 - true_class, tensor_log(negative_class_probability))
    );
}