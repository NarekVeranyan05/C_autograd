//
// Created by Narek Veranian on 2026-07-24.
//

#ifndef UNTITLED_TOPOLOGICAL_SORT_H
#define UNTITLED_TOPOLOGICAL_SORT_H

#include "tensor.h"

/**
 * Derives a topological sorting of the computational graph of the loss function.
 * In a topological sort, a node appears after all of its dependencies
 * @param loss the function to sort topologically
 * @return the nodes in the DAG for the loss, sorted topologically from loss to the beginning
 * @note the returned linked list, when destructed, will not free the data within its nodes
 */
struct Linked_list *topological_sort(struct Tensor *loss);

#endif //UNTITLED_TOPOLOGICAL_SORT_H