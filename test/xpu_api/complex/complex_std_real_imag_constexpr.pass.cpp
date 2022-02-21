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
// class TestStdRealImagConstexpr - testing of constexpr std::real, constexpr std::imag
// 
// described https://en.cppreference.com/w/cpp/numeric/complex
//
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestStdRealImagConstexpr
{
public:

    // https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
    //      C++11 : __cplusplus is 201103L.
    //      C++14 : __cplusplus is 201402L.
    //      C++20 : __cplusplus is 202002L.

    TestStdRealImagConstexpr(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    void run_test()
    {
        // https://en.cppreference.com/w/cpp/numeric/complex/real2
        // until C++14
        //      template< class T >
        //      T real(const std::complex<T>& z);
        // since C++14
        //      template< class T >
        //      constexpr T real(const std::complex<T>& z);
        // ------------
        // since C++11 until C++14
        //      float real( float z );
        //      template< class DoubleOrInteger >
        //      double real(DoubleOrInteger z);
        //      long double real(long double z);
        // since C++14
        //      constexpr float real(float z);
        //      template< class DoubleOrInteger >
        //      constexpr double real(DoubleOrInteger z);
        //      constexpr long double real(long double z);
#if __cplusplus >= 201402L
        {
            constexpr dpl::complex<float> complex_val(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
            constexpr float val = dpl::real(complex_val);
            EXPECT_TRUE_EE(errorEngine, val == COMPLEX_REAL_PART, "Wrong effect from dpl::real #1");
        }

        {
            constexpr float f = 2.3f;
            constexpr float val = dpl::real(f);
            EXPECT_TRUE_EE(errorEngine, val == f, "Wrong effect from dpl::real #2");
        }

        my_invoke_if<IsSupportedDouble>()(
            [&]()
            {
                constexpr double d = 2.3;
                constexpr double val = dpl::real(d);
                EXPECT_TRUE_EE(errorEngine, val == d, "Wrong effect from dpl::real #3");
            });

        my_invoke_if<IsSupportedLongDouble>()(
            [&]()
            {
                constexpr long double d = 2.3;
                constexpr long double val = dpl::real(d);
                EXPECT_TRUE_EE(errorEngine, val == d, "Wrong effect from dpl::real #3");
            });
#endif // __cplusplus >= 201402L

        // https://en.cppreference.com/w/cpp/numeric/complex/imag2
        // until C++14
        //      template< class T >
        //      T imag(const std::complex<T>& z);
        // since C++14
        //      template< class T >
        //      constexpr T imag(const std::complex<T>& z);
        // ------------
        // since C++11 until C++14
        //      float imag(float z);
        //      template< class DoubleOrInteger >
        //      double imag(DoubleOrInteger z);
        //      long double imag(long double z);
        // since C++14
        //      constexpr float imag(float z);
        //      template< class DoubleOrInteger >
        //      constexpr double imag(DoubleOrInteger z);
        //      constexpr long double imag(long double z);
#if __cplusplus >= 201402L
        {
            constexpr dpl::complex<float> complex_val(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
            constexpr float val = dpl::real(complex_val);
            EXPECT_TRUE_EE(errorEngine, val == COMPLEX_REAL_PART, "Wrong effect from dpl::imag #1");
        }

        {
            constexpr float f = 2.3f;
            constexpr float val = dpl::imag(f);
            EXPECT_TRUE_EE(errorEngine, 0 == val, "Wrong effect from dpl::imag #2");
        }

        my_invoke_if<IsSupportedDouble>()(
            [&]()
            {
                constexpr double d = 2.3;
                constexpr double val = dpl::imag(d);
                EXPECT_TRUE_EE(errorEngine, 0 == val, "Wrong effect from dpl::imag #3");
            });

        my_invoke_if<IsSupportedLongDouble>()(
            [&]()
            {
                constexpr long double d = 2.3;
                constexpr long double val = dpl::imag(d);
                EXPECT_TRUE_EE(errorEngine, 0 == val, "Wrong effect from dpl::real #3");
            });
#endif // __cplusplus >= 201402L
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
                            TestStdRealImagConstexpr<ErrorEngine_KernelPart_Impl, ::std::true_type, ::std::false_type> tcc(error_engine_kernel_part);
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
                            TestStdRealImagConstexpr<ErrorEngine_KernelPart_Impl, ::std::false_type, ::std::false_type> tcc(error_engine_kernel_part);
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
    TestStdRealImagConstexpr<TestUtils::ErrorEngineHost, ::std::true_type, ::std::true_type> tcc(error_engine_host);
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
