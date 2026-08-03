//
// Created by Narek Veranian on 2026-07-24.
//

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "topological_sort.h"
#include "linked_list.h"
#include "backprop.h"

volatile int successes = 0;
volatile int failures = 0;

void test_topological_sort_singleton()
{
    int shape[2] = {1, 1};
    struct Tensor *loss = create_tensor(shape, 2, NULL, 0, OP_UNDEF);

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
    struct Tensor *w1= create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    struct Tensor *w2= create_tensor(shape, 2, NULL, 0, OP_UNDEF);
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

void test_scalar_backprop_deep()
{
    int shape[2] = {1, 1};
    struct Tensor *w1= create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    struct Tensor *w2= create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    w1->data[0] = 2;
    w2->data[0] = 3;

    struct Tensor *v1= tensor_log(w1);

    struct Tensor *v2 = tensor_mul(w2, v1);

    struct Tensor *v3 = tensor_scale(3, v2);

    struct Tensor *loss = tensor_add(v3, v2);
    backprop(loss);

    struct Tensor *grad_v3_expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    grad_v3_expected->data[0] = 1;

    struct Tensor *grad_v2_expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    grad_v2_expected->data[0] = 4;

    struct Tensor *grad_v1_expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    grad_v1_expected->data[0] = 4 * w2->data[0];

    struct Tensor *grad_w1_expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    grad_w1_expected->data[0] = 5.99970;

    struct Tensor *grad_w2_expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    grad_w2_expected->data[0] = 2.772588;

    int count = 0;
    count += check_equal(grad_v3_expected, v3->grad);
    count += check_equal(grad_v2_expected, v2->grad);
    count += check_equal(grad_v1_expected, v1->grad);
    count += check_equal(grad_w1_expected, w1->grad);
    count += check_equal(grad_w2_expected, w2->grad);

    if (count == 5)
    {
        printf("SUCCESS: backprop evaluates gradients as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: backprop failed.\n");
        failures++;
    }

    tensor_destruct(grad_v3_expected, true);
    tensor_destruct(grad_v2_expected, true);
    tensor_destruct(grad_v1_expected, true);
    tensor_destruct(grad_w1_expected, true);
    tensor_destruct(grad_w2_expected, true);
}


int main()
{
    test_topological_sort_singleton();
    test_topological_sort_deep();

    test_scalar_backprop_deep();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}
