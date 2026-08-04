//
// Created by Narek Veranian on 2026-07-24.
//

#include <assert.h>
#include <stdlib.h>

#include "./stack.h"
#include "./linked_list.h"

struct Stack {
    struct Linked_list *ll;
};

struct Stack *create_stack()
{
    struct Stack *stack = malloc(sizeof(struct Stack));
    stack->ll = create_linked_list();

    return stack;
}

void destruct_stack(struct Stack *stack)
{
    assert(stack != NULL);

    destruct_linked_list(stack->ll);
    free(stack);
}

void push(const struct Stack *stack, void *data, void (*destruct_data)(void *data))
{
    add_at(stack->ll, 0, data, destruct_data);
}

void *peek(const struct Stack *stack)
{
    return get_at(stack->ll, 0);
}

void pop(const struct Stack *stack)
{
    remove_at(stack->ll, 0);
}

bool search(const struct Stack *stack, void *data)
{
   return contains(stack->ll, data);
}


bool isEmpty(const struct Stack *stack)
{
    return (size(stack->ll) == 0);
}