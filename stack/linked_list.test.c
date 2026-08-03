//
// Created by Narek Veranian on 2026-07-24.
//

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "linked_list.c"

volatile int successes = 0;
volatile int failures = 0;

void test_size()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);
    remove_at(ll, 0);

    if (ll->size == 3)
    {
        printf("SUCCESS: linked list size updates as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked list size is inconsistent.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

void test_add()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);


    const struct Node *curr_node = ll->head;
    int count = 0;
    for (int i = 0; i < ll->size; i++)
    {
        count += (*(int *)curr_node->data == x[i]);
        curr_node = curr_node->next;
    }

    if (count == 4)
    {
        printf("SUCCESS: linked list add returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked list add failed.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

void test_remove_head()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);
    remove_at(ll, 0);

    const struct Node *curr_node = ll->head;
    int count = 0;
    for (int i = 0; i < ll->size; i++)
    {
        count += (*(int *)curr_node->data == x[i+1]);
        curr_node = curr_node->next;
    }

    if (count == 3)
    {
        printf("SUCCESS: linked list removes head as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked list remove head failed.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

void test_remove_mid()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);
    remove_at(ll, 1);

    const struct Node *curr_node = ll->head;
    int count = 0;
    for (int i = 0; i < ll->size; i++)
    {
        if (i >= 1)
            count += (*(int *)curr_node->data == x[i+1]);
        else
            count += (*(int *)curr_node->data == x[i]);
        curr_node = curr_node->next;
    }

    if (count == 3)
    {
        printf("SUCCESS: linked list removes intermediate node as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked list remove intermediate node failed.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

void test_remove_end()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);
    remove_at(ll, 3);

    const struct Node *curr_node = ll->head;
    int count = 0;
    for (int i = 0; i < ll->size; i++)
    {
        count += (*(int *)curr_node->data == x[i]);
        curr_node = curr_node->next;
    }

    if (count == 3)
    {
        printf("SUCCESS: linked list removes tail as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked list remove tail failed.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

void test_remove_all()
{
    struct Linked_list *ll = create_linked_list();
    int x[4] = {1, 2, 3, 4};
    add(ll, &x[2], NULL);
    add_at(ll, 0, &x[0], NULL);
    add_at(ll, 1, &x[1], NULL);
    add(ll, &x[3], NULL);
    remove_all(ll);

    if (ll->size == 0 && ll->head == NULL)
    {
        printf("SUCCESS: linked list removes all nodes as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: linked remove all nodes failed.\n");
        failures++;
    }

    destruct_linked_list(ll);
}

int main()
{
    test_size();

    test_add();

    test_remove_head();
    test_remove_mid();
    test_remove_end();

    test_remove_all();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}