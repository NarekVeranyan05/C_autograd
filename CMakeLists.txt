cmake_minimum_required(VERSION 3.20)
project(untitled C)

set(CMAKE_C_STANDARD 11)

include_directories(
        backward
        linalg
        tensor
)

# Only build the test executable containing main() from tensor.test.c
add_executable(tensor_test
        tensor/tensor.test.c
        tensor/tensor.c
        tensor/tensor.h
        backward/backward.c
        backward/backward.h
        linalg/linalg.h
)