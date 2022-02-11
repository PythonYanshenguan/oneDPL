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
#define COMPLEX_IMAG_PART_ZERO 0.0

template <typename TComplexDataType>
void
test_host()
{
    // Check complex(real)
    {
        const auto complex_val = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART);
        EXPECT_TRUE(COMPLEX_REAL_PART == complex_val.real(), "Wrong effect of dpl::complex::real()");
        EXPECT_TRUE(COMPLEX_IMAG_PART_ZERO == complex_val.imag(), "Wrong effect of dpl::complex::imag()");

        EXPECT_TRUE(dpl::complex<TComplexDataType>(COMPLEX_REAL_PART) == complex_val, "Wrong effect of dpl::complex::operator==");
    }

    // Check complex(real, imag)
    {
        const auto complex_val = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART);

        EXPECT_TRUE(COMPLEX_REAL_PART == complex_val.real(), "Wrong effect of dpl::complex::real()");
        EXPECT_TRUE(COMPLEX_IMAG_PART == complex_val.imag(), "Wrong effect of dpl::complex::imag()");

        EXPECT_TRUE(dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART) == complex_val, "Wrong effect of dpl::complex::operator==");
    }
}

#if TEST_DPCPP_BACKEND_PRESENT
template <typename TComplexDataType>
void
test_kernel(sycl::queue& deviceQueue)
{
    using ComplexDataType = dpl::complex<TComplexDataType>;

    cl::sycl::cl_bool ret[2] = { true, true };
    const cl::sycl::range<1> numOfItems{ 2 };

    ComplexDataType host_complex_data[] = { dpl::complex<TComplexDataType>(COMPLEX_REAL_PART),
                                            dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART) };

    {
        cl::sycl::buffer<ComplexDataType, 1> host_buffer(&host_complex_data[0], numOfItems);
        cl::sycl::buffer<cl::sycl::cl_bool, 1> ret_buffer(&ret[0], numOfItems);

        deviceQueue.submit(
            [&](cl::sycl::handler& cgh)
            {
                auto accessor_to_host_buffer = host_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
                auto accessor_ret = ret_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);

                cgh.single_task<TComplexDataType>(
                    [=]()
                    {
                        // Check complex(real)
                        {
                            ComplexDataType& device_complex_data = accessor_to_host_buffer[0];

                            const bool real_part_eq = COMPLEX_REAL_PART == device_complex_data.real();
                            accessor_ret[0] &= (real_part_eq);

                            const bool imag_part_eq = COMPLEX_IMAG_PART_ZERO == device_complex_data.imag();
                            accessor_ret[0] &= (imag_part_eq);

                            const bool complex_eq = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART) == device_complex_data;
                            accessor_ret[0] &= (complex_eq);
                        }

                        // Check complex(real, imag)
                        {
                            ComplexDataType& device_complex_data = accessor_to_host_buffer[1];

                            const bool real_part_eq = COMPLEX_REAL_PART == device_complex_data.real();
                            accessor_ret[1] &= (real_part_eq);

                            const bool imag_part_eq = COMPLEX_IMAG_PART == device_complex_data.imag();
                            accessor_ret[1] &= (imag_part_eq);

                            const bool complex_eq = dpl::complex<TComplexDataType>(COMPLEX_REAL_PART, COMPLEX_IMAG_PART) == device_complex_data;
                            accessor_ret[1] &= (complex_eq);
                        }
                    });
            });
    }

    EXPECT_TRUE(ret[0], "Wrong effect of dpl::complex(real) : real or imag");
    EXPECT_TRUE(ret[1], "Wrong effect of dpl::complex(real, imag) : real or imag");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    test_host<float>();
    test_host<double>();
    test_host<long double>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };
        const auto& device = deviceQueue.get_device();

        // https://en.cppreference.com/w/cpp/header/complex
        // The specializations std::complex<float>, std::complex<double>, and std::complex<long double>
        // are LiteralTypes for representing and manipulating complex numbers.
        // The effect of instantiating the template complex for any other type is unspecified.
        // Implementations may forbid instantiation of such specializations.
        test_kernel<float>(deviceQueue);

        if (device.has(sycl::aspect::fp64))
            test_kernel<double>(deviceQueue);

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        //test_kernel<long double>(deviceQueue);
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
