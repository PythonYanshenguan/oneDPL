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

#include <cmath>
#include <math.h>

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>
#endif

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;
#endif

#include "cases.h"

#define COMPLEX_REAL_PART 1.5
#define COMPLEX_IMAG_PART 2.25
#define EPSILON           0.000001

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
// class TestComplexAbs - testing of std::abs from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/abs :
//      template< class T >
//      T abs(const complex<T>& z);
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexAbs
{
public:

    TestComplexAbs(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    void run_test()
    {
        test_abs<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        my_invoke_if<IsSupportedDouble>()([&](){ test_abs<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        my_invoke_if<IsSupportedLongDouble>()([&](){ test_abs<long double>(); });

        // Test cases from libxcxx checks
        my_invoke_if<IsSupportedDouble>()([&]() { test_edges(); });
    }

protected:

    template <typename TComplexDataType>
    void test_abs()
    {
        const auto complex_val = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
        const auto abs_res = dpl::abs(complex_val);

        auto real_part = complex_val.real();
        auto imag_part = complex_val.imag();
        auto abs_res_expected = sqrt(real_part * real_part + imag_part * imag_part);

        EXPECT_TRUE_EE(errorEngine, dpl::abs(abs_res - abs_res_expected) < EPSILON, "Wrong result if dpl::abs(dpl::complex<T>()) function");
    }

    void test_edges()
    {
        const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
        for (unsigned i = 0; i < N; ++i)
        {
            double r = dpl::abs(testcases[i]);
            switch (classify(testcases[i]))
            {
            case zero:
                EXPECT_TRUE_EE(errorEngine, r == 0, "Wrong result if dpl::abs(dpl::complex<T>()) function #1");
                EXPECT_TRUE_EE(errorEngine, !std::signbit(r), "Wrong result if dpl::abs(dpl::complex<T>()) function #2");
                break;
            case non_zero:
                EXPECT_TRUE_EE(errorEngine, std::isfinite(r) && r > 0, "Wrong result if dpl::abs(dpl::complex<T>()) function #3");
                break;
            case inf:
                EXPECT_TRUE_EE(errorEngine, std::isinf(r) && r > 0, "Wrong result if dpl::abs(dpl::complex<T>()) function #4");
                break;
            case NaN:
                EXPECT_TRUE_EE(errorEngine, std::isnan(r), "Wrong result if dpl::abs(dpl::complex<T>()) function #5");
                break;
            case non_zero_nan:
                EXPECT_TRUE_EE(errorEngine, std::isnan(r), "Wrong result if dpl::abs(dpl::complex<T>()) function #6");
                break;
            }
        }
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

        //if (has_type_support<double>(deviceQueue.get_device()))
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
                            TestComplexAbs<ErrorEngine_KernelPart_Impl, ::std::true_type, ::std::false_type> tcc(error_engine_kernel_part);
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
                            TestComplexAbs<ErrorEngine_KernelPart_Impl, ::std::false_type, ::std::false_type> tcc(error_engine_kernel_part);
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
    //TestComplexAbs<TestUtils::ErrorEngineHost, ::std::true_type, ::std::true_type> tcc(error_engine_host);
    //tcc.run_test();

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

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
