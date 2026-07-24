//
// Created by Narek Veranian on 2026-07-24.
//

#ifndef UNTITLED_LINKED_LIST_HPP
#define UNTITLED_LINKED_LIST_HPP

struct Node {
    void *data;
    void (*destruct_data)(void *data);
    struct Node *next;
};

struct Linked_list {
    struct Node *head;
    int size;
};

struct Linked_list *create_linked_list();
void destruct_linked_list(struct Linked_list* ll);
void *get_at(const struct Linked_list *ll, int index);
void add(struct Linked_list *ll, void *data, void (*destruct_data)(void *data));
void add_at(struct Linked_list *ll, int index, void *data, void (*destruct_data)(void *data));
void remove_all(struct Linked_list *ll);
void remove_at(struct Linked_list *ll, int index);

#endif //UNTITLED_LINKED_LIST_HPP
