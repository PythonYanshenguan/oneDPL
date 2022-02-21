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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/complex>

#include "support/test_config.h"
#include "support/utils.h"
#include "support/utils_err_eng.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>
#endif

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;
#endif

#define COMPLEX_REAL_PART 1.5
#define COMPLEX_IMAG_PART 2.25

template <typename TRequiredType, typename TVal>
void check_type(TVal val)
{
    static_assert(::std::is_same<typename ::std::decay<TVal>::type, TRequiredType>::value, "Types should be equals");
}

namespace
{
    template <typename IsOperationSupported>
    struct my_invoke_if
    {
        template <typename Op/*, typename... Rest*/>
        void
        operator()(Op op/*, Rest&&... rest*/)
        {
            //op(::std::forward<Rest>(rest)...);
            op();
        }
    };

    template <>
    struct my_invoke_if<::std::false_type>
    {
        template <typename Op/*, typename... Rest*/>
        void
        operator()(Op op/*, Rest&&... rest*/)
        {
            // Do not call op;
        }
    };
};


////////////////////////////////////////////////////////////////////////////////
// class TestComplexRealImagConstexpr - testing of std::conj from <complex>
// 
// described https://en.cppreference.com/w/cpp/numeric/complex
//
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexRealImagConstexpr
{
public:

    // https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
    //      C++11 : __cplusplus is 201103L.
    //      C++20 : __cplusplus is 202002L.

    TestComplexRealImagConstexpr(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    void run_test()
    {
        test<float>();

        my_invoke_if<IsSupportedDouble>()([&]() { test<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        my_invoke_if<IsSupportedLongDouble>()([&]() { test<long double>(); });
    }

protected:

    template <class T>
    void test_constexpr()
    {
#if __cplusplus >= 201103L
        constexpr std::complex<T> c1;
        EXPECT_TRUE_EE(errorEngine, c1.real() == 0, "Wrong effect of conj in real part #1");
        EXPECT_TRUE_EE(errorEngine, c1.imag() == 0, "Wrong effect of conj in imag part #1");

        constexpr std::complex<T> c2(3);
        EXPECT_TRUE_EE(errorEngine, c2.real() == 3, "Wrong effect of conj in real part #2");
        EXPECT_TRUE_EE(errorEngine, c2.imag() == 0, "Wrong effect of conj in imag part #2");

        constexpr std::complex<T> c3(3, 4);
        EXPECT_TRUE_EE(errorEngine, c3.real() == 3, "Wrong effect of conj in real part #3");
        EXPECT_TRUE_EE(errorEngine, c3.imag() == 4, "Wrong effect of conj in imag part #3");
#endif // __cplusplus >= 201103L
    }

    template <class T>
    void test()
    {
        std::complex<T> c;
        EXPECT_TRUE_EE(errorEngine, c.real() == 0, "Wrong effect of conj in real part #4");
        EXPECT_TRUE_EE(errorEngine, c.imag() == 0, "Wrong effect of conj in imag part #4");

        c.real(3.5);
        EXPECT_TRUE_EE(errorEngine, c.real() == 3.5, "Wrong effect of conj in real part #5");
        EXPECT_TRUE_EE(errorEngine, c.imag() == 0, "Wrong effect of conj in imag part #5");

        c.imag(4.5);
        EXPECT_TRUE_EE(errorEngine, c.real() == 3.5, "Wrong effect of conj in real part #6");
        EXPECT_TRUE_EE(errorEngine, c.imag() == 4.5, "Wrong effect of conj in imag part #6");

        c.real(-4.5);
        EXPECT_TRUE_EE(errorEngine, c.real() == -4.5, "Wrong effect of conj in real part #7");
        EXPECT_TRUE_EE(errorEngine, c.imag() == 4.5, "Wrong effect of conj in imag part #7");

        c.imag(-5.5);
        EXPECT_TRUE_EE(errorEngine, c.real() == -4.5, "Wrong effect of conj in real part #8");
        EXPECT_TRUE_EE(errorEngine, c.imag() == -5.5, "Wrong effect of conj in imag part #8");

        test_constexpr<T>();
    }

private:

    TErrorEngine& errorEngine;
};

#if TEST_DPCPP_BACKEND_PRESENT
void
test_kernel(sycl::queue& deviceQueue)
{
    TestUtils::ErrorEngine_HostPart error_engine_host_part;

    const auto& device = deviceQueue.get_device();

    {
        auto sycl_buf_host_errors = error_engine_host_part.get_sycl_buffer();

        if (device.has(sycl::aspect::fp64))
        {
            deviceQueue.submit(
                [&](cl::sycl::handler& cgh)
                {
                    auto accessor_to_sycl_buf_host_errors = sycl_buf_host_errors.template get_access<cl::sycl::access::mode::read_write>(cgh);

                    cgh.single_task<class TestKernel1>(
                        [=]()
                        {
                            // Prepare kernel part of error engine
                            using ErrorEngine_KernelPart_Impl = ::TestUtils::ErrorEngine_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                            ErrorEngine_KernelPart_Impl error_engine_kernel_part(accessor_to_sycl_buf_host_errors);

                            // Run test in kernel
                            TestComplexRealImagConstexpr<ErrorEngine_KernelPart_Impl, ::std::true_type, ::std::false_type> tcc(error_engine_kernel_part);
                            tcc.run_test();
                        });
                });
        }
        else
        {
            deviceQueue.submit(
                [&](cl::sycl::handler& cgh)
                {
                    auto accessor_to_sycl_buf_host_errors = sycl_buf_host_errors.template get_access<cl::sycl::access::mode::read_write>(cgh);

                    cgh.single_task<class TestKernel2>(
                        [=]()
                        {
                            // Prepare kernel part of error engine
                            using ErrorEngine_KernelPart_Impl = ::TestUtils::ErrorEngine_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                            ErrorEngine_KernelPart_Impl error_engine_kernel_part(accessor_to_sycl_buf_host_errors);

                            // Run test in kernel
                            TestComplexRealImagConstexpr<ErrorEngine_KernelPart_Impl, ::std::false_type, ::std::false_type> tcc(error_engine_kernel_part);
                            tcc.run_test();
                        });
                });
        }
    }

    error_engine_host_part.process_errors();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    // Prepare host error engine
    TestUtils::ErrorEngineHost error_engine_host;

    // Run test on host
    TestComplexRealImagConstexpr<TestUtils::ErrorEngineHost, ::std::true_type, ::std::true_type> tcc(error_engine_host);
    tcc.run_test();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        test_kernel(deviceQueue);
    }
    catch (const std::exception& exc)
    {
        std::string errorMsg = "Exception occurred";
        if (exc.what())
        {
            errorMsg += " : ";
            errorMsg += exc.what();
        }

        EXPECT_TRUE(false, errorMsg.c_str());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
