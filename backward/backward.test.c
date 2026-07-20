//
// Created by Narek Veranian on 2026-07-20.
//

#include <time.h>
#include <stdlib.h>
#include <sys/_types/_null.h>

#include "./tensor.h"
#include <stdio.h>

#include "backward.h"

volatile int successes;
volatile int failures;

void test_scalar_backward_sum()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, MATMUL);
    v1->data[0] = 1;
    v2->data[0] = 1;

    struct Tensor *v3 = tensor_add(v1, v2);
    init_grad(v3);
    v3->grad->data[0] = 5;
    v3->backward = backward_sum;

    v3->backward(v3);

    if (check_equal(v1->grad, v3->grad) && check_equal(v2->grad, v3->grad))
    {
        printf("SUCCESS: scalar backward sum updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward sum grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
}

void test_vec_backward_sum()
{
    int shape[2] = {3, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, MATMUL);
    v1->data[0] = 1, v1->data[1] = 2, v1->data[2] = 3;
    v2->data[0] = 1, v2->data[1] = 2, v2->data[2] = 3;

    struct Tensor *v3 = tensor_add(v1, v2);
    init_grad(v3);
    v3->grad->data[0] = 5, v3->grad->data[1] = 7.75, v3->grad->data[2] = -10;
    v3->backward = backward_sum;

    v3->backward(v3);

    if (check_equal(v1->grad, v3->grad) && check_equal(v2->grad, v3->grad))
    {
        printf("SUCCESS: vector backward sum updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector backward sum grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
}

int main() {
    test_scalar_backward_sum();
    // test_vec_backward_sum();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}