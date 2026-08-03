//
// Created by Narek Veranian on 2026-07-20.
//

#include <time.h>
#include <stdlib.h>
#include <sys/_types/_null.h>

#include "./tensor.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "backward.h"

volatile int successes = 0;
volatile int failures = 0;

void test_scalar_backward_sum()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
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

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
}

void test_vec_backward_sum()
{
    int shape[2] = {3, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
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

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
}

void test_scalar_backward_matmul()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1->data[0] = 8;
    v2->data[0] = 5;

    struct Tensor *v3 = tensor_matmul(v1, v2);
    init_grad(v3);
    v3->grad->data[0] = 3;
    v3->backward = backward_matmul;

    v3->backward(v3);

    struct Tensor *v1_grad_expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2_grad_expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1_grad_expected->data[0] = 15;
    v2_grad_expected->data[0] = 24;

    if (check_equal(v1_grad_expected, v1->grad) && check_equal(v2_grad_expected, v2->grad))
    {
        printf("SUCCESS: scalar backward OP_MATMUL updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward OP_MATMUL grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
    tensor_destruct(v1_grad_expected, true);
    tensor_destruct(v2_grad_expected, true);
}

void test_vec_backward_matmul_1()
{
    int shape1[2] = {3, 1};
    int shape2[2] = {1, 3};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);
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
        printf("SUCCESS: vector backward OP_MATMUL (resulting in a matrix) updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector backward OP_MATMUL (resulting in a matrix) grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
    tensor_destruct(v1_grad_expected, true);
    tensor_destruct(v2_grad_expected, true);
}

void test_vec_backward_matmul_2()
{
    int shape1[2] = {2, 3};
    int shape2[2] = {3, 1};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);
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
        printf("SUCCESS: scalar backward OP_MATMUL updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward OP_MATMUL grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
    tensor_destruct(v1_grad_expected, true);
    tensor_destruct(v2_grad_expected, true);
}

void test_matrix_backward_matmul()
{
    int shape1[2] = {2, 3};
    int shape2[2] = {3, 2};
    struct Tensor *v1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *v2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

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
        printf("SUCCESS: matrix backward OP_MATMUL updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: matrix backward OP_MATMUL grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(v3, true);
    tensor_destruct(v1_grad_expected, true);
    tensor_destruct(v2_grad_expected, true);
}

void test_scalar_backward_sigmoid()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1->data[0] = 5;

    struct Tensor *v2 = tensor_sigmoid(v1);
    init_grad(v2);
    v2->grad->data[0] = 3;
    v2->backward = backward_sigmoid;

    v2->backward(v2);

    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    expected->data[0] = 3 * exp(-5) / pow(1 + exp(-5), 2);

    if (check_equal(expected, v1->grad))
    {
        printf("SUCCESS: scalar backward sigmoid updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward sigmoid grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(expected, true);
}

void test_vector_backward_sigmoid()
{
    int shape[2] = {3, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1->data[0] = 5, v1->data[1] = 6, v1->data[2] = -5;

    struct Tensor *v2 = tensor_sigmoid(v1);
    init_grad(v2);
    v2->grad->data[0] = 3, v2->grad->data[1] = 4, v2->grad->data[2] = 5;
    v2->backward = backward_sigmoid;

    v2->backward(v2);

    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    expected->data[0] = 3 * exp(-5) / pow(1 + exp(-5), 2);
    expected->data[1] = 4 * exp(-6) / pow(1 + exp(-6), 2);
    expected->data[2] = 5 * exp(5) / pow(1 + exp(5), 2);

    if (check_equal(expected, v1->grad))
    {
        printf("SUCCESS: vector backward sigmoid updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector backward sigmoid grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(expected, true);
}

void test_scalar_backward_ReLU1()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1->data[0] = 8;

    struct Tensor *v2 = tensor_ReLU(v1);
    init_grad(v2);
    v2->grad->data[0] = 3;
    v2->backward = backward_ReLU;

    v2->backward(v2);

    struct Tensor *exp = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    exp->data[0] = 3;

    if (check_equal(exp, v1->grad))
    {
        printf("SUCCESS: scalar backward ReLU updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward ReLU grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(exp, true);
}

void test_scalar_backward_ReLU2()
{
    int shape[2] = {1, 1};
    struct Tensor *v1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    v1->data[0] = -8;

    struct Tensor *v2 = tensor_ReLU(v1);
    init_grad(v2);
    v2->grad->data[0] = 3;
    v2->backward = backward_ReLU;

    v2->backward(v2);

    struct Tensor *exp = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    exp->data[0] = 0;

    if (check_equal(exp, v1->grad))
    {
        printf("SUCCESS: scalar backward ReLU updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar backward ReLU grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v1, true);
    tensor_destruct(v2, true);
    tensor_destruct(exp, true);
}

void test_backward_softmax()
{
    int shape[2] = {3, 1};
    struct Tensor *v = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    v->data[0] = 1, v->data[1] = 2, v->data[2] = -2;
    struct Tensor *s = tensor_softmax(v);
    s->backward = backward_softmax;

    struct Tensor *label_prob = tensor_selector(s, 0);
    label_prob->backward = backward_matmul;

    init_grad(label_prob);
    label_prob->grad->data[0] = 1;
    label_prob->backward(label_prob);
    s->backward(s);

    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_UNDEF);
    expected->data[0] = 0.1949571818;
    expected->data[1] = -0.1914506406;
    expected->data[2] = -0.0035065408;

    if (check_equal(expected, v->grad))
    {
        printf("SUCCESS: backward softmax updates grads expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: backward softmax grad updates failed.\n");
        failures++;
    }

    tensor_destruct(v, true);
    tensor_destruct(s, true);
    tensor_destruct(label_prob, true);
    tensor_destruct(expected, true);
}

int main() {
    test_scalar_backward_sum();
    test_vec_backward_sum();

    test_scalar_backward_matmul();
    test_vec_backward_matmul_1();
    test_vec_backward_matmul_2();
    test_matrix_backward_matmul();

    test_scalar_backward_sigmoid();
    test_vector_backward_sigmoid();

    test_scalar_backward_ReLU1();
    test_scalar_backward_ReLU2();

    test_backward_softmax();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}