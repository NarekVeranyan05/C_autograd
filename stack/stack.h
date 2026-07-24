//
// Created by Narek Veranian on 2026-07-24.
//

#ifndef UNTITLED_STACK_H
#define UNTITLED_STACK_H

struct Stack;

struct Stack *create_stack();
void destruct_stack(struct Stack *stack);
void push(const struct Stack *stack, void *data, void (*destruct_data)(void *data));
void *peek(const struct Stack *stack);
void pop(const struct Stack *stack);

#endif //UNTITLED_STACK_H
