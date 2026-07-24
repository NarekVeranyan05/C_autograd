//
// Created by Narek Veranian on 2026-07-24.
//

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "topological_sort.h"
#include "linked_list.h"

volatile int successes = 0;
volatile int failures = 0;

void test_topological_sort_singleton()
{
    int shape[2] = {1, 1};
    struct Tensor *loss = create_tensor(shape, 2, NULL, 0, UNDEF);

    struct Linked_list *sorted = topological_sort(loss);
    if (size(sorted) == 1 && get_at(sorted, 0) == loss)
    {
        printf("SUCCESS: topological sort is in expected order.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: topological sort failed.\n");
        failures++;
    }

    destruct_linked_list(sorted);
}

void test_topological_sort_deep()
{
    int shape[2] = {1, 1};
    struct Tensor *w1= create_tensor(shape, 2, NULL, 0, UNDEF);
    struct Tensor *w2= create_tensor(shape, 2, NULL, 0, UNDEF);
    w1->data[0] = 2;
    w2->data[0] = 3;

    struct Tensor *v1= tensor_log(w1);

    struct Tensor *v2 = tensor_mul(w2, v1);

    struct Tensor *v3 = tensor_scale(3, v2);

    struct Tensor *loss = tensor_add(v3, v2);

    struct Linked_list *sorted = topological_sort(loss);


    struct Tensor *expected_order[7] = {loss, v3, NULL, v2, w2, v1, w1};

    if (size(sorted) == 7)
    {
        printf("SUCCESS: topological sort has expected length.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: topological sort length failed.\n");
        failures++;
    }

    int count = 0;
    for (int i = 0; i < 7; i++)
    {
        if (expected_order[i] == NULL) continue;
        count += ( get_at(sorted, i) == expected_order[i] );
    }

    if (count == 6)
    {
        printf("SUCCESS: topological order is as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: topological sort order failed.\n");
        failures++;
    }

    destruct_linked_list(sorted);
}

int main()
{
    // test_topological_sort_singleton();
    test_topological_sort_deep();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}
