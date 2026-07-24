//
// Created by Narek Veranian on 2026-07-24.
//

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "tensor.h"
#include "linked_list.h"
#include "topological_sort.h"

void backprop(struct Tensor *loss) {
    assert(loss != NULL);

    // initialize self-gradient as 1
    init_grad(loss);
    loss->grad->data[0] = 1;

    struct Linked_list *sorted = topological_sort(loss);

    // pass gradients backward
    while (size(sorted) != 0)
    {
        struct Tensor *curr_node = get_at(sorted, 0);
        if (curr_node->backward != NULL)
            curr_node->backward(curr_node);

        remove_at(sorted, 0);
    }
}
