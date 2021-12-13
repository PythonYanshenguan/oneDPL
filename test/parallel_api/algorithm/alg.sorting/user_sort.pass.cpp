// -*- C++ -*-
//===-- user_sort.pass.cpp ------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"
#endif

#if TEST_DPCPP_BACKEND_PRESENT

template <sycl::usm::alloc alloc_type>
void
test_with_usm(sycl::queue& q)
{
    constexpr int N = 32;

    int h_key[N] = {};
    int h_val[N] = {};
    for (int i = 0; i < N; i++)
    {
        h_val[i] = ((N - 1 - i) / 3) * 3;
        h_key[i] = i * 10;
    }

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_key(q, ::std::begin(h_key), ::std::end(h_key));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(q, ::std::begin(h_val), ::std::end(h_val));

    int* d_key = dt_helper_h_key.get_data();
    int* d_val = dt_helper_h_val.get_data();

    auto first = oneapi::dpl::make_zip_iterator(d_key, d_val);
    auto last = first + std::distance(d_key, d_key + N);

    auto myPolicy = oneapi::dpl::execution::make_device_policy<
        TestUtils::unique_kernel_name<class copy, (::std::size_t)alloc_type>>(q);
    std::sort(myPolicy, first, last,
              [](const auto& it1, const auto& it2)
              {
                  using std::get;
                  return get<0>(it1) > get<0>(it2);
              });

    int h_skey[N] = {};
    int h_sval[N] = {};

    dt_helper_h_key.retrieve_data(h_skey);
    dt_helper_h_val.retrieve_data(h_sval);

    for (int i = 0; i < N; i++)
    {
        if (i < (N - 1))
        {
            EXPECT_TRUE(h_skey[i] >= h_skey[i + 1], "wrong sort result");
        }
    }
}

#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto exception_handler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const& e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e)
            {
                std::cout << "Caught asynchronous SYCL exception during calculation:\n" << e.what() << std::endl;
            }
        }
    };

    sycl::queue q(exception_handler);
    std::cout << "    Device Name = " << q.get_device().get_info<cl::sycl::info::device::name>().c_str() << "\n";

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared>(q);
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device>(q);
#endif

    return TestUtils::done();
}
