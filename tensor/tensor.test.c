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

volatile int successes = 0;
volatile int failures = 0;

void test_scalar_tensor_add()
{
    int shape[2] = {1, 1};
    struct Tensor *t1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    t1->data[0] = 3;
    t2->data[0] = 4.5;
    expected->data[0] = 7.5;

    struct Tensor *res = tensor_add(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: scalar addition returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar addition failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_multidims_tensor_add()
{
    int shape1[3] = {3, 3, 2}, shape2[3] = {3, 3, 2};
    struct Tensor *t1 = create_tensor(shape1, 3, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape2, 3, NULL, 0, OP_MATMUL);

    const double d1[18] = {
        1, -1, 4,
        0, 3, 2,
        2, 1, 0,

        1, 5, -3,
        7, 8, 11,
        -20, 8, 90
    };
    memcpy(t1->data, d1, 9 * sizeof(double));
    const double d2[18] = {
        3, 1, 2,
        2, 1, 0,
        1, 4, 5,

        88, 12, 9,
        33, -41, 22,
        1, 23, 0
    };
    memcpy(t2->data, d2, 9 * sizeof(double));

    int shape[3] = {3, 3, 2};
    struct Tensor *expected = create_tensor(shape, 3, NULL, 0, OP_MATMUL);
    double const exp[18] = {
        4, 0, 6,
        2, 4, 2,
        3, 5, 5,

        89, 17, 6,
        40, -33, 33,
        -19, 31, 90
    };
    memcpy(expected->data, exp, 9 * sizeof(double));

    struct Tensor *res = tensor_add(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: tensor addition returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: tensor addition failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_scalar_matmul()
{
    int shape[2] = {1, 1};
    struct Tensor *t1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    t1->data[0] = 3;
    t2->data[0] = 4.5;
    expected->data[0] = 13.5;

    struct Tensor *res = tensor_matmul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: scalar matmul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar matmul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_vec_matmul()
{
    int shape1[2] = {1, 3}, shape2[2] = {3, 1};
    struct Tensor *t1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

    t1->data[0] = 3, t1->data[1] = 4, t1->data[2] = 5;
    t2->data[0] = 5, t2->data[1] = -4, t2->data[2] = 0;

    int shape[2] = {1, 1};
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    expected->data[0] = -1;

    struct Tensor *res = tensor_matmul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: vector matmul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector matmul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_square_matmul()
{
    int shape1[2] = {3, 3}, shape2[2] = {3, 3};
    struct Tensor *t1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

    const double d1[9] = {1, -1, 4, 0, 3, 2, 2, 1, 0};
    memcpy(t1->data, d1, 9 * sizeof(double));
    const double d2[9] = {3, 1, 2, 2, 1, 0, 1, 4, 5};
    memcpy(t2->data, d2, 9 * sizeof(double));

    int shape[2] = {3, 3};
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    double const exp[9] = {5, 16, 22, 8, 11, 10, 8, 3, 4};
    memcpy(expected->data, exp, 9 * sizeof(double));

    struct Tensor *res = tensor_matmul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: square matrix matmul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: square matrix matmul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_non_square_matmul()
{
    int shape1[2] = {3, 4}, shape2[2] = {4, 5};
    struct Tensor *t1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

    const double d1[12] = {0, 0, 7, 2, 4, 1, 8, 1, 9, 1, 4, 0};
    memcpy(t1->data, d1, 12 * sizeof(double));
    const double d2[20] = {1, 26, 12, 11, -6, 1, 0, 8, 6, -1, -5, -4, 1, 2, 6, 7, 9, 12, 0, -24};
    memcpy(t2->data, d2, 20 * sizeof(double));

    int shape[2] = {3, 5};
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    const double exp[15] = {-21, -10, 31, 14, -6, -28, 81, 76, 66, -1, -10, 218, 120, 113, -31};
    memcpy(expected->data, exp, 15 * sizeof(double));

    struct Tensor *res = tensor_matmul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: non-square matmul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: non-square matmul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_tensor_scalar_mul()
{
    int shape[2] = {1, 1};
    struct Tensor *t1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    t1->data[0] = 3;
    t1->data[1] = 4.5;

    struct Tensor *res = tensor_mul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: scalar mul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar mul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_vec_mul()
{
    int shape[2] = {1, 3};
    struct Tensor *t1 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    t1->data[0] = 3, t1->data[1] = 4, t1->data[2] = 5;
    t2->data[0] = 5, t2->data[1] = -4, t2->data[2] = 0;

    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    expected->data[0] = 15, expected->data[1] = -16, expected->data[2] = 0;

    struct Tensor *res = tensor_mul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: vector mul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector mul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_matrix_mul() {
    int shape1[2] = {3, 5}, shape2[2] = {3, 5};
    struct Tensor *t1 = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *t2 = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

    const double d1[15] = {1, -1, 4, 0, 3, 2, 2, 1, 0, 7, 1, -52, 11, 5, 8};
    memcpy(t1->data, d1,15  * sizeof(double));
    const double d2[15] = {3, 1, 2, 2, 1, 0, 1, 4, 5, 6, 21, -11, 0, -15, 1};
    memcpy(t2->data, d2, 15 * sizeof(double));

    int shape[2] = {3, 5};
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    double const exp[15] = {3, -1, 8, 0, 3, 0, 2, 4, 0, 42, 21, 572, 0, -75, 8};
    memcpy(expected->data, exp, 15 * sizeof(double));

    struct Tensor *res = tensor_mul(t1, t2);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: square matrix matmul returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: square matrix matmul failed.\n");
        failures++;
    }

    tensor_destruct(t1, true);
    tensor_destruct(t2, true);
    tensor_destruct(expected, true);
    tensor_destruct(res, true);
}

void test_scalar_transpose()
{
    int shape[2] = {1, 1};
    struct Tensor *t = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    t->data[0] = 4;
    expected->data[0] = 4;

    struct Tensor *res = tensor_transpose(t);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: scalar transpose returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: scalar transpose failed.\n");
        failures++;
    }

    free(t->data);
    free(res->data);
    free(expected->data);
    free(t);
    free(res);
    free(expected);
}

void test_vector_transpose()
{
    int shape1[2] = {1, 4};
    int shape2[2] = {4, 1};
    struct Tensor *t = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);
    t->data[0] = expected->data[0] = 1, t->data[1] = expected->data[1] = 2, t->data[2] = expected->data[2] = 3, t->data[3] = expected->data[3] = 4;

    struct Tensor *res = tensor_transpose(t);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: vector transpose returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: vector transpose failed.\n");
        failures++;
    }

    tensor_destruct(t, true);
    tensor_destruct(res, true);
    tensor_destruct(expected, true);
}

void test_matrix_transpose()
{
    int shape1[2] = {3, 5};
    int shape2[2] = {5, 3};
    struct Tensor *t = create_tensor(shape1, 2, NULL, 0, OP_MATMUL);
    struct Tensor *expected = create_tensor(shape2, 2, NULL, 0, OP_MATMUL);

    const double d[15] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };
    memcpy(t->data, d, 15 * sizeof(double));

    const double exp[15] = {
        1, 6, 11,
        2, 7, 12,
        3, 8, 13,
        4, 9, 14,
        5, 10, 15
    };
    memcpy(expected->data, exp, 15 * sizeof(double));


    struct Tensor *res = tensor_transpose(t);
    if (check_equal(expected, res))
    {
        printf("SUCCESS: matrix transpose returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: matrix transpose failed.\n");
        failures++;
    }

    tensor_destruct(t, true);
    tensor_destruct(expected, true);
}

extern const double epsilon;

void test_tensor_invert()
{
    int shape[2] = {3, 5};
    struct Tensor *t = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    const double d[15] = {
        1, -5, 0, 11, -20,
        6, 7, -30, -12, 10,
        11, 12, 0, -1, -15
    };
    memcpy(t->data, d, 15 * sizeof(double));

    struct Tensor *t_inverted_exp = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    const double exp[15] = {
        (1 / (epsilon + 1)), (1/ (epsilon - 5)), (1/epsilon), (1 / (epsilon + 11)), (1 / (epsilon - 20)),
        (1 / (epsilon + 6)), (1 / (epsilon + 7)), (1 / (epsilon - 30)), (1 / (epsilon - 12)), (1 / (epsilon + 10)),
        (1 / (epsilon + 11)), (1 / (epsilon + 12)), (1 / epsilon), (1 / (epsilon - 1)), (1 / (epsilon - 15))
    };
    memcpy(t_inverted_exp->data, exp, 15 * sizeof(double));

    struct Tensor *t_inverted = tensor_invert(t);

    if (check_equal(t_inverted, t_inverted_exp))
    {
        printf("SUCCESS: inversion returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: inversion failed.\n");
        failures++;
    }

    tensor_destruct(t, true);
    tensor_destruct(t_inverted_exp, true);
    tensor_destruct(t_inverted, true);
}

void test_tensor_ReLU()
{
    int shape[2] = {3, 5};
    struct Tensor *t = create_tensor(shape, 2, NULL, 0, OP_MATMUL);

    const double d[15] = {
        1, -5, 0, 11, -20,
        6, 7, -30, -12, 10,
        11, 12, 0, -1, -15
    };
    memcpy(t->data, d, 15 * sizeof(double));

    struct Tensor *t_activated_exp = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    const double exp[15] = {
        1, 0, 0, 11, 0,
        6, 7, 0, 0, 10,
        11, 12, 0, 0, 0
    };
    memcpy(t_activated_exp->data, exp, 15 * sizeof(double));

    struct Tensor *t_activated = tensor_ReLU(t);

    if (check_equal(t_activated, t_activated_exp))
    {
        printf("SUCCESS: ReLU returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: ReLU failed.\n");
        failures++;
    }

    tensor_destruct(t, true);
    tensor_destruct(t_activated_exp, true);
    tensor_destruct(t_activated, true);
}

void test_vector_softmax()
{
    int shape[2] = {5, 1};
    struct Tensor *t = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    t->data[0] = 1;
    t->data[1] = -1;
    t->data[2] = 0;
    t->data[3] = 2;
    t->data[4] = -2;

    struct Tensor *expected = create_tensor(shape, 2, NULL, 0, OP_MATMUL);
    expected->data[0] = 0.234122;
    expected->data[1] = 0.031685;
    expected->data[2] = 0.086129;
    expected->data[3] = 0.636409;
    expected->data[4] = 0.011656;

    struct Tensor *softmax = tensor_softmax(t);
    for (int i = 0; i < 5; i++)
        softmax->data[i] = ceil(softmax->data[i] * 1000000.0) / 1000000.0;

    if (check_equal(softmax, expected))
    {
        printf("SUCCESS: Softmax returns as expected.\n");
        successes++;
    }
    else
    {
        printf("FAILURE: Softmax failed.\n");
        failures++;
    }

    tensor_destruct(t, true);
    tensor_destruct(expected, true);
    tensor_destruct(softmax, true);
}

int main() {
    test_scalar_tensor_add();
    test_multidims_tensor_add();

    test_scalar_matmul();
    test_vec_matmul();
    test_square_matmul();
    test_non_square_matmul();

    test_tensor_scalar_mul();
    test_vec_mul();
    test_matrix_mul();

    test_scalar_transpose();
    test_vector_transpose();
    test_matrix_transpose();

    test_tensor_invert();

    test_tensor_ReLU();

    test_vector_softmax();

    printf("\nTotal tests: %d\n", successes + failures);
    printf("Total successes: %d\n", successes);
    printf("Total failures: %d\n", failures);

    return EXIT_SUCCESS;
}