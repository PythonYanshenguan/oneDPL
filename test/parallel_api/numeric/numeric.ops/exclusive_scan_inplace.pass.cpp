// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#define _GLIBCXX_USE_TBB_PAR_BACKEND 0 // libstdc++10

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

#include <CL/sycl.hpp>
//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/algorithm>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

#include <cassert>
#include <vector>
#include <iostream>
#include <functional>

template <typename T>
void dump_data(const char* msg, const T& data)
{
    std::cout << msg;
    for (auto item : data)
        std::cout << item << " ";
    std::cout << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_with_usm(sycl::queue& q)
{
    std::vector<int> v{ 3, 1, 4, 1, 5, 9, 2, 6 };

    dump_data("Source data: ", v);

    std::vector<int> serial_exclusive_scan_result(v.size());
    exclusive_scan_serial(v.begin(), v.end(), serial_exclusive_scan_result.begin(), 0);

    dump_data("Serial exlusive scan result: ", serial_exclusive_scan_result);

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper(q, v.begin(), v.end());
    int* excl_input_dev = dt_helper.get_data();

    // STL Exclusive scan (in-place, correct results)
    std::exclusive_scan(v.begin(), v.end(), v.begin(), 0);

    dump_data("std::exclusive_scan result: ", v);

    {
        TestUtils::usm_data_transfer<alloc_type, int> dt_helper_res(q, v.begin(), v.end());
        int* excl_input_dev_res = dt_helper_res.get_data();
        oneapi::dpl::exclusive_scan(oneapi::dpl::execution::make_device_policy(q), excl_input_dev, excl_input_dev + v.size(), excl_input_dev_res, 0);

        std::vector<int> parallel_exclusive_scan_result(v.size());
        dt_helper_res.retrieve_data(parallel_exclusive_scan_result.begin());

        dump_data("oneapi::dpl::exclusive_scan result: ", parallel_exclusive_scan_result);
    }

    // DPL Exclusive scan (in-place, incorrect results)
    oneapi::dpl::exclusive_scan(oneapi::dpl::execution::make_device_policy(q), excl_input_dev, excl_input_dev + v.size(), excl_input_dev, 0);

    std::vector<int> excl_result_host_data_vector(v.size(), 0);
    dt_helper.retrieve_data(excl_result_host_data_vector.begin());

    dump_data("oneapi::dpl::exclusive_scan inplace result: ", excl_result_host_data_vector);

    for (int i = 0; i < v.size(); i++)
    {
        assert(v[i] == excl_result_host_data_vector[i]);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q;
#if _ONEDPL_DEBUG_SYCL
    std::cout << "    Device Name = " << q.get_device().get_info<cl::sycl::info::device::name>().c_str() << "\n";
#endif // _ONEDPL_DEBUG_SYCL

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared>(q);
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device>(q);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}