//
// Created by Narek Veranian on 2026-08-03.
//

#ifndef UNTITLED_QUEUE_H
#define UNTITLED_QUEUE_H

#include <stdbool.h>

struct Queue;

struct Queue *create_queue();
void destruct_queue(struct Queue *queue);

void enqueue(const struct Queue *queue, void *data, void (*destruct_data)(void *data));
void *peek_head(const struct Queue *queue);
void *peek_tail(const struct Queue *queue);
void *dequeue(const struct Queue *queue);

bool isEmpty(const struct Queue *queue);

#endif // UNTITLED_QUEUE_H
