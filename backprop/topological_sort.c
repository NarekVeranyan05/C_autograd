//
// Created by Narek Veranian on 2026-07-24.
//

#include "topological_sort.h"

#include <stdlib.h>

#include "linked_list.h"
#include "stack.h"
#include "tensor.h"

static void tensor_destruct_void(void *tensor)
{
    tensor_destruct(tensor);
}

struct Linked_list *topological_sort(struct Tensor *loss)
{

    struct Linked_list *sorted = create_linked_list();

    // pushing the loss to the stack
    struct Stack *stack = create_stack();
    push(stack, loss, NULL);

    while (!isEmpty(stack))
    {
        struct Tensor *curr_node = peek(stack);

        // if current node is not visited, mark it as visited
        if (!contains(sorted, curr_node))
            add(sorted, curr_node, tensor_destruct_void);

        bool found_new = false; // not visited or not pushed
        for (int i = 0; i < curr_node->num_parents && !found_new; i++)
        {
            found_new = !contains(sorted, curr_node->parents[i]) && !search(stack, curr_node->parents[i]);

            if (found_new)
                push(stack, curr_node->parents[i], NULL);
        }

        if (!found_new)
            pop(stack);
    }

    destruct_stack(stack);

    return sorted;
}

