// -*- C++ -*-
//===-- gather_if.pass.cpp ----------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// Tests for gather_if

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#include <random> //std::default_random_engine

struct is_even {
    bool operator()(int x) const {
        return (x % 2) == 0;
    }
};

template <typename Policy, typename InputIter1, typename InputIter2, typename InputIter3, typename OutputIter, typename Predicate>
OutputIter gather_if(Policy&& policy, InputIter1 map_first, InputIter1 map_last, InputIter2 mask, InputIter3 input_first, OutputIter result, Predicate pred) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input_first, map_first);
    const int n = std::distance(map_first, map_last);

    return oneapi::dpl::transform_if(policy, perm_begin, perm_begin + n, mask, result,
        [=](auto&& v){ return v; }, [=](auto&& m){ return pred(m); }
    );
}

void test_gather_if(int input_size) {
    sycl::queue q;

    auto sycl_deleter = [q](int* mem) { sycl::free(mem, q.get_context()); };

    ::std::unique_ptr<int, decltype(sycl_deleter)>

    data((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    indices((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    mask((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    result((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter);

    int* data_ptr = data.get();
    int* idx_ptr = indices.get();
    int* mask_ptr = mask.get();
    int* res_ptr = result.get();

    for (int i = 0; i != input_size; ++i) {
        data_ptr[i] = i+1; // data = {1, 2, 3, ..., n}
        idx_ptr[i] = i; // indices = {0, 1, 2, ..., n-1}
        mask_ptr[i] = (i+1) % 2; // mask = {1, 0, 1, 0, ..., 1, 0}
        res_ptr[i] = 0; // result = {0, 0, 0, ..., 0}
    }

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // randomize map and mask indices
    shuffle(idx_ptr, idx_ptr + input_size, std::default_random_engine(seed));
    shuffle(mask_ptr, mask_ptr + input_size, std::default_random_engine(seed));

    // call gather_if
    gather_if(oneapi::dpl::execution::dpcpp_default, idx_ptr, idx_ptr + input_size, mask_ptr, data_ptr, res_ptr, oneapi::dpl::identity());

    q.wait_and_throw();

    // test if gather_if has correct output
    for (int i = 0; i != input_size; ++i) {
        if ((mask_ptr[i] == 1 && res_ptr[i] != data_ptr[idx_ptr[i]]) || (mask_ptr[i] == 0 && res_ptr[i] != 0)) {
            std::cout << "Input size " << input_size << ": Failed\n";
            break;
        }
    }
}

void test_gather_if_with_pred(int input_size) {
    sycl::queue q;

    auto sycl_deleter = [q](int* mem) { sycl::free(mem, q.get_context()); };

    ::std::unique_ptr<int, decltype(sycl_deleter)>

    data((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    indices((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    mask((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    result((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter);

    int* data_ptr = data.get();
    int* idx_ptr = indices.get();
    int* mask_ptr = mask.get();
    int* res_ptr = result.get();

    for (int i = 0; i != input_size; ++i) {
        data_ptr[i] = i+1; // data = {1, 2, 3, ..., n}
        idx_ptr[i] = i; // indices = {0, 1, 2, ..., n-1}
        mask_ptr[i] = i*3; // mask = {0, 3, 6, ..., 3(n-1)}
        res_ptr[i] = 0; // result = {0, 0, 0, ..., 0}
    }

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // randomize map and mask indices
    shuffle(idx_ptr, idx_ptr + input_size, std::default_random_engine(seed));
    shuffle(mask_ptr, mask_ptr + input_size, std::default_random_engine(seed));

    // call gather_if
    gather_if(oneapi::dpl::execution::dpcpp_default, idx_ptr, idx_ptr + input_size, mask_ptr, data_ptr, res_ptr, is_even());

    q.wait_and_throw();

    // test if gather_if has correct output
    for (int i = 0; i != input_size; ++i) {
        if ((mask_ptr[i] % 2 == 0 && res_ptr[i] != data_ptr[idx_ptr[i]]) || (mask_ptr[i] % 2 == 1 && res_ptr[i] != 0)) {
            std::cout << "Input size " << input_size << ": Failed\n";
            break;
        }
    }
}

int main() {
    const int max_n = 100000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        test_gather_if(n);
        test_gather_if_with_pred(n);
    }

    return TestUtils::done();
}
