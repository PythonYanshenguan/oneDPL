//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class T>
// struct iterator_traits<const T*>

#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>
#include "test_macros.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

struct A
{
};

void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef s::iterator_traits<const volatile A*> It;
            static_assert((s::is_same<It::difference_type, s::ptrdiff_t>::value), "");
            static_assert((s::is_same<It::pointer, const volatile A*>::value), "");
            static_assert((s::is_same<It::reference, const volatile A&>::value), "");
            static_assert((s::is_same<It::iterator_category, s::random_access_iterator_tag>::value), "");
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