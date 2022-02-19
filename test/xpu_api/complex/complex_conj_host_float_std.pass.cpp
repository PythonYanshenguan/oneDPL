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

#include <oneapi/dpl/complex>

#include "support/utils.h"

int
main()
{
    int is_done = 0;

#if __cplusplus >= 201103L && __cplusplus < 202002L
    {
        // https://en.cppreference.com/w/cpp/numeric/complex/conj
        // since C++11
        // until C++20
        //      std::complex<float> conj(float z);
        //      template< class DoubleOrInteger >
        //      std::complex<double> conj(DoubleOrInteger z);
        //      std::complex<long double> conj(long double z);

        float z = (float)2.3f;
        auto complex_val_res = ::std::conj(z);

        // https://github.com/oneapi-src/oneDPL/runs/5257802314?check_suite_focus=true
        // error: member reference base type 'float' is not a structure or union
        std::cout << "real : " << complex_val_res.real() << std::endl;
        // https://github.com/oneapi-src/oneDPL/runs/5257802314?check_suite_focus=true
        // error: member reference base type 'float' is not a structure or union
        std::cout << "imag : " << complex_val_res.imag() << std::endl;

        is_done = 1;
    }
#endif

    return TestUtils::done(is_done);
}
