cmake_minimum_required(VERSION 3.20)
project(untitled C)

set(CMAKE_C_STANDARD 11)

include_directories(
        backward
        tensor
        stack
        backprop
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
        backprop/topological_sort.c
        backprop/topological_sort.h
)

# 2. Executable for Stack tests
add_executable(stack_test
        stack/stack.h
        stack/linked_list.h
        stack/linked_list.c
        stack/stack_ll.c
        stack/linked_list.test.c
)

# 3. Executable for Topological sort tests
add_executable(topological_sort_test
        backward/backward.c
        backward/backward.h
        tensor/tensor.c
        tensor/tensor.h
        stack/stack.h
        stack/linked_list.h
        stack/linked_list.c
        stack/stack_ll.c
        backprop/backprop.c
        backprop/topological_sort.c
        backprop/topological_sort.h
        backprop/topological_sort.test.c
)