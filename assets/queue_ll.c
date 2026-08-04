//
// Created by Narek Veranian on 2026-08-03.
//

#include <assert.h>
#include <stdlib.h>

#include "linked_list.c"
#include "queue.h"

struct Queue {
    struct Linked_list *ll;
};

struct Queue *create_queue()
{
    struct Queue *q = malloc(sizeof(struct Queue));

    q->ll = create_linked_list();

    return q;
}

void destruct_queue(struct Queue *queue)
{
    assert(queue != NULL);

    destruct_linked_list(queue->ll);
    free(queue);
}

void enqueue(const struct Queue *queue, void *data, void (*destruct_data)(void *data))
{
    assert(queue != NULL);

    add(queue->ll, data, destruct_data);
}

void *peek_head(const struct Queue *queue)
{
    assert(queue != NULL);

    return get_at(queue->ll, 0);
}

void *peek_tail(const struct Queue *queue)
{
    assert(queue != NULL);

    return get_at(queue->ll, size(queue->ll) - 1);
}

void *dequeue(const struct Queue *queue)
{
    assert(queue != NULL);

    void *data = peek_head(queue);
    remove_at(queue->ll, 0);

    return data;
}

bool isEmpty(const struct Queue *queue)
{
    assert(queue != NULL);

    return (size(queue->ll) == 0);
}

int main() {
    int var1 = 1;
    int var2 = 2;
    int var3 = 3;
    int var4 = 4;

    struct Queue *q = create_queue();
    enqueue(q, &var1, NULL);
    enqueue(q, &var2, NULL);
    enqueue(q, &var3, NULL);
    enqueue(q, &var4, NULL);

    assert(isEmpty(q) == false);
    assert(peek_head(q) == &var1);
    assert(peek_tail(q) == &var4);
    assert(dequeue(q) == &var1);
    assert(dequeue(q) == &var2);
    assert(dequeue(q) == &var3);
    assert(dequeue(q) == &var4);
    assert(isEmpty(q));

    return EXIT_SUCCESS;
}