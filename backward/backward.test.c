//
// Created by Narek Veranian on 2026-07-20.
//

#include <time.h>
#include <stdlib.h>
#include <sys/_types/_null.h>

#include "./tensor.h"
#include <stdio.h>
#include <string.h>

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

void test_scalar_backward_matmul()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, MATMUL);
    v1->data[0] = 8;
    v2->data[0] = 5;

    struct Tensor *v3 = tensor_matmul(v1, v2);
    init_grad(v3);
    v3->grad->data[0] = 3;
    v3->backward = backward_matmul;

    v3->backward(v3);

    struct Tensor *v1_grad_expected = create_tensor(shape, 2, NULL, 0, MATMUL);
    struct Tensor *v2_grad_expected = create_tensor(shape, 2, NULL, 0, MATMUL);
    v1_grad_expected->data[0] = 15;
    v2_grad_expected->data[0] = 24;

    if (check_equal(v1_grad_expected, v1->grad) && check_equal(v2_grad_expected, v2->grad))
    {
        printf("SUCCESS: scalar backward matmul updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward matmul grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
    tensor_destruct(v1_grad_expected);
    tensor_destruct(v2_grad_expected);
}

void test_vec_backward_matmul_1()
{
    int shape1[2] = {3, 1};
    int shape2[2] = {1, 3};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, MATMUL);
    v1->data[0] = 8, v1->data[1] = 4, v1->data[2] = -5;
    v2->data[0] = 11, v2->data[1] = 0, v2->data[2] = 3;

    struct Tensor *v3 = tensor_matmul(v1, v2);
    init_grad(v3);
    double temp[9] = {
        3, 4, 5,
        -5, 10, 6,
        11, 3, 0
    };
    memcpy(v3->grad->data, temp, sizeof(double) * 9);
    v3->backward = backward_matmul;

    v3->backward(v3);

    struct Tensor *v1_grad_expected = tensor_matmul(v3->grad, tensor_transpose(v2));
    struct Tensor *v2_grad_expected = tensor_matmul(tensor_transpose(v1), v3->grad);

    if (check_equal(v1_grad_expected, v1->grad) && check_equal(v2_grad_expected, v2->grad))
    {
        printf("SUCCESS: vector backward matmul (resulting in a matrix) updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector backward matmul (resulting in a matrix) grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
    tensor_destruct(v1_grad_expected);
    tensor_destruct(v2_grad_expected);
}

void test_vec_backward_matmul_2()
{
    int shape1[2] = {2, 3};
    int shape2[2] = {3, 1};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, MATMUL);
    v1->data[0] = 8, v1->data[1] = 4, v1->data[2] = -5;
    v2->data[0] = 11, v2->data[1] = 0, v2->data[2] = 3;

    struct Tensor *v3 = tensor_matmul(v1, v2);
    init_grad(v3);
    v3->grad->data[0] = 3;
    v3->backward = backward_matmul;

    v3->backward(v3);

    struct Tensor *v1_grad_expected = tensor_matmul(v3->grad, tensor_transpose(v2));
    struct Tensor *v2_grad_expected = tensor_matmul(tensor_transpose(v1), v3->grad);

    if (check_equal(v1_grad_expected, v1->grad) && check_equal(v2_grad_expected, v2->grad))
    {
        printf("SUCCESS: scalar backward matmul updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward matmul grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
    tensor_destruct(v1_grad_expected);
    tensor_destruct(v2_grad_expected);
}

void test_matrix_backward_matmul()
{
    int shape1[2] = {2, 3};
    int shape2[2] = {3, 2};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, MATMUL);

    const double d1[6] = {
        1, 2, 3,
        4, 5, 6
    };
    memcpy(v1->data, d1, 6 * sizeof(double));

    double d2[6] = {
        7, 8,
        9, 10,
        11, 12
    };
    memcpy(v2->data, d2, 6 * sizeof(double));

    struct Tensor *v3 = tensor_matmul(v1, v2);
    init_grad(v3);

    double d3[4] = {
        1, 1,
        1, 1
    };
    memcpy(v3->grad->data, d3, 6 * sizeof(double));
    v3->backward = backward_matmul;

    v3->backward(v3);

    struct Tensor *v1_grad_expected = tensor_matmul(v3->grad, tensor_transpose(v2));
    struct Tensor *v2_grad_expected = tensor_matmul(tensor_transpose(v1), v3->grad);

    if (check_equal(v1_grad_expected, v1->grad) && check_equal(v2_grad_expected, v2->grad))
    {
        printf("SUCCESS: matrix backward matmul updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: matrix backward matmul grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1);
    tensor_destruct(v2);
    tensor_destruct(v3);
    tensor_destruct(v1_grad_expected);
    tensor_destruct(v2_grad_expected);
}

int main() {
    test_scalar_backward_sum();
    test_vec_backward_sum();

    test_scalar_backward_matmul();
    test_vec_backward_matmul_1();
    test_vec_backward_matmul_2();
    test_matrix_backward_matmul();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}