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
// class TestComplexConj - testing of std::conj from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/conj :
// (1)
//      (until C++20)
//          template< class T >
//          std::complex<T> conj(const std::complex<T>& z);
//      (since C++20)
//          template< class T >
//          constexpr std::complex<T> conj(const std::complex<T>& z);
// (2)
//      (since C++11)
//      (until C++20)
//          std::complex<float> conj(float z);
//          template< class DoubleOrInteger >
//          std::complex<double> conj(DoubleOrInteger z);
//          std::complex<long double> conj(long double z);
//      (since C++20)
//          constexpr std::complex<float> conj(float z);
//          template< class DoubleOrInteger >
//          constexpr std::complex<double> conj(DoubleOrInteger z);
//          constexpr std::complex<long double> conj(long double z);
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexConj
{
public:

    // https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
    //      C++11 : __cplusplus is 201103L.
    //      C++20 : __cplusplus is 202002L.

    TestComplexConj(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    void run_test()
    {
        test_form_1<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        my_invoke_if<IsSupportedDouble>()([&](){ test_form_1<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        my_invoke_if<IsSupportedLongDouble>()([&](){ test_form_1<long double>(); });

#if __cplusplus >= 202002L

        test_form_1_since_CPP20<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        my_invoke_if<IsSupportedDouble>()([&](){ test_form_1_since_CPP20<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        my_invoke_if<IsSupportedLongDouble>()([&]() { test_form_1_since_CPP20<long double>(); });

#endif // __cplusplus >= 202002L

#if __cplusplus >= 201103L
        test_form_2_since_CPP11();
#if __cplusplus >= 202002L
        test_form_2_since_CPP20();
#endif // __cplusplus >= 202002L
#endif // __cplusplus >= 201103L
    }

protected:

    // Test form (1) untin C++20
    //          template< class T >
    //          std::complex<T> conj(const std::complex<T>& z);
    template <typename TComplexDataType>
    void test_form_1()
    {
        auto complex_val_src = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
        auto complex_val_res = dpl::conj(complex_val_src);
        EXPECT_TRUE_EE(errorEngine, complex_val_src.real() == complex_val_res.real(), "Wrong effect of conj in real part");
        EXPECT_TRUE_EE(errorEngine, complex_val_src.imag() == complex_val_res.imag() * -1, "Wrong effect of conj in imag part");
    }

#if __cplusplus >= 202002L
    // Test form (1) since C++20
    //          template< class T >
    //          constexpr std::complex<T> conj(const std::complex<T>& z);
    template <typename TComplexDataType>
    void test_form_1_since_CPP20()
    {
        // const
        {
            auto complex_val_src = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
            auto complex_val_res = dpl::conj(complex_val_src);
            EXPECT_TRUE_EE(errorEngine, complex_val_src.real() == complex_val_res.real(), "Wrong effect of conj in real part #1");
            EXPECT_TRUE_EE(errorEngine, complex_val_src.imag() == complex_val_res.imag() * -1, "Wrong effect of conj in imag part #1");
        }

        // constexpr
        {
            constexpr dpl::complex<TComplexDataType> complex_val_src(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);
            constexpr dpl::complex<TComplexDataType> complex_val_res(dpl::conj(complex_val_src));
            EXPECT_TRUE_EE(errorEngine, complex_val_src.real() == complex_val_res.real(), "Wrong effect of conj in real part #2");
            EXPECT_TRUE_EE(errorEngine, complex_val_src.imag() == complex_val_res.imag() * -1, "Wrong effect of conj in imag part #2");
        }
    }
#endif // __cplusplus >= 202002L

#if __cplusplus >= 201103L
    //      (since C++11)
    //      (until C++20)
    //          1) std::complex<float> conj(float z);
    //          2) template< class DoubleOrInteger >
    //             std::complex<double> conj(DoubleOrInteger z);
    //          3) std::complex<long double> conj(long double z);
    void test_form_2_since_CPP11()
    {
        // 1) std::complex<float> conj(float z);
        test_form_2_since_CPP11_for_type<float>(2.3f);
        //{
        //    float z = 2.3;
        //    auto complex_val_res = dpl::conj(z);
        //    check_type<float>(complex_val_res.real());
        //    check_type<float>(complex_val_res.imag());
        //    EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #3");
        //    EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #3");
        //}

        // 2) template< class DoubleOrInteger >
        //    std::complex<double> conj(DoubleOrInteger z);
        {
            // double
            my_invoke_if<IsSupportedDouble>()([&](){ test_form_2_since_CPP11_for_type<double>(2.3); });

            // Commented, because in DPCPP this code not compiled: error: member reference base type 'float' is not a structure or union
            //                                                     check_type<double>(complex_val_res.real());
            //                                                                                        ^
            //// integer
            //my_invoke_if<IsSupportedDouble>()([&](){ test_form_2_since_CPP11_for_type<int, double>(2); });
        }

        // 3) std::complex<long double> conj(long double z);
        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        my_invoke_if<IsSupportedLongDouble>()([&](){ test_form_2_since_CPP11_for_type<long double>(2.3); });
    }

    template <typename TComplexDataType, typename TRequiredComplexFieldsType = TComplexDataType>
    void
    test_form_2_since_CPP11_for_type(TComplexDataType initVal)
    {
        TComplexDataType z = (TComplexDataType)initVal;
        auto complex_val_res = dpl::conj(z);
        check_type<TRequiredComplexFieldsType>(complex_val_res.real());
        check_type<TRequiredComplexFieldsType>(complex_val_res.imag());
        EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #3");
        EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #3");
    }

#endif // __cplusplus >= 201103L

#if __cplusplus >= 202002L
    //      (since C++20)
    //          1) constexpr std::complex<float> conj(float z);
    //          2) template< class DoubleOrInteger >
    //             constexpr std::complex<double> conj(DoubleOrInteger z);
    //          3) constexpr std::complex<long double> conj(long double z);
    void test_form_2_since_CPP20()
    {
        // 1) std::complex<float> conj(float z);
        {
            constexpr float z = 2.3;
            constexpr auto complex_val_res = dpl::conj(z);
            check_type<float>(complex_val_res.real());
            check_type<float>(complex_val_res.imag());
            EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #4");
            EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #4");
        }

        // 2) template< class DoubleOrInteger >
        //    std::complex<double> conj(DoubleOrInteger z);
        {
            // double
            my_invoke_if<IsSupportedDouble>()(
                [&]()
                {
                    constexpr double z = 2.3;
                    constexpr auto complex_val_res = dpl::conj(z);
                    check_type<double>(complex_val_res.real());
                    check_type<double>(complex_val_res.imag());
                    EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #5");
                    EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #5");
                });

            // Commented, because in DPCPP this code not compiled: error: member reference base type 'float' is not a structure or union
            //                                                     check_type<double>(complex_val_res.real());
            //                                                                                        ^
            //// integer
            //my_invoke_if<IsSupportedDouble>()(
            //    [&]()
            //    {
            //        constexpr int z = 2;
            //        constexpr auto complex_val_res = dpl::conj(z);
            //        check_type<double>(complex_val_res.real());
            //        check_type<double>(complex_val_res.imag());
            //        EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #6");
            //        EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #6");
            //    });
        }

        // 3) std::complex<long double> conj(long double z);
        my_invoke_if<IsSupportedLongDouble>()(
            [&]()
            {
                constexpr long double z = 2.3;
                constexpr auto complex_val_res = dpl::conj(z);
                check_type<long double>(complex_val_res.real());
                check_type<long double>(complex_val_res.imag());
                EXPECT_TRUE_EE(errorEngine, complex_val_res.real() == z, "Wrong effect of conj in real part #7");
                EXPECT_TRUE_EE(errorEngine, complex_val_res.imag() == 0.0, "Wrong effect of conj in imag part #7");
            });
    }
#endif // __cplusplus >= 202002L

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
                            TestComplexConj<ErrorEngine_KernelPart_Impl, ::std::true_type, ::std::false_type> tcc(error_engine_kernel_part);
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
                            TestComplexConj<ErrorEngine_KernelPart_Impl, ::std::false_type, ::std::false_type> tcc(error_engine_kernel_part);
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
    TestComplexConj<TestUtils::ErrorEngineHost, ::std::true_type, ::std::true_type> tcc(error_engine_host);
    tcc.run_test();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        test_kernel(deviceQueue);
    }
    catch (const sycl::exception& e)
    {
        std::string errorMsg;
        errorMsg = "\t\tSYCL exception during generation\n";
        errorMsg += e.what();
        errorMsg += "\n";
        errorMsg += "OpenCL status: ";
        errorMsg += e.get_cl_code();
        errorMsg += "\n";

        EXPECT_TRUE(false, errorMsg.c_str());
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
