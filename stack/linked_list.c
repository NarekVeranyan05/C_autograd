//
// Created by Narek Veranian on 2026-07-24.
//

#include "linked_list.h"

#include <assert.h>
#include <stdlib.h>

struct Node {
    void *data;
    void (*destruct_data)(void *data);
    struct Node *next;
};

struct Linked_list {
    struct Node *head;
    int size;
};

struct Linked_list *create_linked_list()
{
    struct Linked_list *ll = malloc(sizeof(struct Linked_list));
    ll->head = NULL;
    ll->size = 0;

    return ll;
}

void destruct_linked_list(struct Linked_list* ll)
{
    remove_all(ll);
    free(ll);
}

void *get_at(const struct Linked_list *ll, const int index)
{
    assert(ll != NULL);

    void *result = NULL;

    if (index >= 0 && index < ll->size)
    {
        struct Node *curr_node = ll->head;
        int curr_index = 0;
        while (curr_index < index)
        {
            curr_node = curr_node->next;
            curr_index++;
        }

        result = curr_node->data;
    }

    return result;
}

bool contains(const struct Linked_list *ll, void *data)
{
    bool contains = false;

    const struct Node *curr_node = ll->head;

    while (curr_node != NULL && !contains)
    {
        contains = (curr_node->data == data);
        curr_node = curr_node->next;
    }

    return contains;
}

void add(struct Linked_list *ll, void *data, void (*destruct_data)(void *data))
{
    assert(ll != NULL);

    struct Node *prev_node = NULL;
    struct Node *curr_node = ll->head;

    while (curr_node != NULL)
    {
        prev_node = curr_node;
        curr_node = curr_node->next;
    }

    if (prev_node == NULL) // if head was null, insert at head
    {
        ll->head = malloc(sizeof(struct Node));
        ll->head->data = data;
        ll->head->destruct_data = destruct_data;
        ll->head->next = NULL;
    }
    else
    {
        prev_node->next = malloc(sizeof(struct Node));
        prev_node->next->data = data;
        prev_node->next->destruct_data = destruct_data;
        prev_node->next->next = NULL;
    }

    ll->size++;
}

void add_at(struct Linked_list *ll, int index, void *data, void (*destruct_data)(void *data))
{
    assert(ll != NULL);

    if (index >= 0)
    {
        struct Node *prev_node = NULL;
        struct Node *curr_node = ll->head;
        int curr_index = 0;

        while (curr_index < index && curr_node != NULL)
        {
            prev_node = curr_node;
            curr_node = curr_node->next;
            curr_index++;
        }

        if (prev_node == NULL) // if index = 0, insert at head
        {
            ll->head = malloc(sizeof(struct Node));
            ll->head->data = data;
            ll->head->destruct_data = destruct_data;
            ll->head->next = curr_node;
        }
        else // if index != 0, insert in-between
        {
            prev_node->next = malloc(sizeof(struct Node));
            prev_node->next->data = data;
            prev_node->next->destruct_data = destruct_data;
            prev_node->next->next = curr_node;
        }

        ll->size++;
    }
}

void remove_all(struct Linked_list *ll)
{
    assert(ll != NULL);
    struct Node *prev_node = NULL;
    struct Node *curr_node = ll->head;

    while (curr_node != NULL)
    {
        prev_node = curr_node;
        curr_node = curr_node->next;

        if (prev_node->destruct_data != NULL)
            prev_node->destruct_data(prev_node->data);
        free(prev_node);
    }

    ll->head = NULL;
    ll->size = 0;
}

void remove_at(struct Linked_list *ll, int index)
{
    assert(ll != NULL);

    if (index >= 0 && index < ll->size)
    {
        struct Node *prev_node = NULL;
        struct Node *curr_node = ll->head;

        if (index == 0)
        {
            ll->head = ll->head->next;

            if (curr_node->destruct_data != NULL)
                curr_node->destruct_data(curr_node->data);
            free(curr_node);
        }
        else
        {
            int curr_index = 0;
            while (curr_index < index)
            {
                prev_node = curr_node;
                curr_node = curr_node->next;

                curr_index++;
            }

            prev_node->next = curr_node->next;
            if (curr_node->destruct_data != NULL)
               curr_node->destruct_data(curr_node->data);
            free(curr_node);
        }

        ll->size--;
    }
}

int size(const struct Linked_list *ll)
{
    return ll->size;
}
