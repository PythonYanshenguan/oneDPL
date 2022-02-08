//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// struct output_iterator_tag {};

#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include "test_macros.h"
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
namespace s = std;
#endif

void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            s::output_iterator_tag tag;
            ((void)tag); // Prevent unused warning
            static_assert((!s::is_base_of<s::input_iterator_tag, s::output_iterator_tag>::value), "");
        });
    });
}

int
main(int, char**)
{
    kernelTest();
    std::cout << "Pass" << std::endl;
    return 0;
}