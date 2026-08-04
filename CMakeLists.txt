cmake_minimum_required(VERSION 3.20)
project(untitled C)

set(CMAKE_C_STANDARD 11)

include_directories(
        backward
        tensor
        assets
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
        backprop/backprop.h
        assets/linked_list.h
        assets/linked_list.c
        assets/stack.h
        assets/stack_ll.c
        model/layers.c
        model/layers.h
        backprop/optimisers/optimiser.h
        backprop/optimisers/sdg.c
        backprop/optimisers/adam.c
        model/model.c
        model/model.h
        assets/queue.h
        assets/queue_ll.c
        model/output_layer.c
        model/loss.h
        model/loss.c
)

# 2. Executable for Stack tests
add_executable(stack_test
        assets/stack.h
        assets/linked_list.h
        assets/linked_list.c
        assets/stack_ll.c
        assets/linked_list.test.c
)

# 3. Executable for Topological sort tests
add_executable(topological_sort_test
        backward/backward.c
        backward/backward.h
        tensor/tensor.c
        tensor/tensor.h
        assets/stack.h
        assets/linked_list.h
        assets/linked_list.c
        assets/stack_ll.c
        backprop/backprop.c
        backprop/topological_sort.c
        backprop/topological_sort.h
        backprop/backprop.test.c
)