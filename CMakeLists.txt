cmake_minimum_required(VERSION 3.20)
project(untitled C)

set(CMAKE_C_STANDARD 11)

include_directories(
        backward
        tensor
        stack
)

# 1. Executable for Tensor tests
add_executable(tensor_test
        tensor/tensor.test.c
        tensor/tensor.c
        tensor/tensor.h
        backward/backward.c
        backward/backward.h
)

# 2. Executable for Backward tests
add_executable(backward_test
        backward/backward.test.c
        backward/backward.c
        backward/backward.h
        tensor/tensor.c
        tensor/tensor.h
        backprop/backprop.c
)

# 2. Executable for Stack tests
add_executable(stack_test
        stack/stack.h
        stack/linked_list.h
        stack/linked_list.c
        stack/stack_ll.c
        stack/linked_list.test.c
)