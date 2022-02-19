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

#if __cplusplus < 202002L
    {
        // https://en.cppreference.com/w/cpp/numeric/complex/conj
        // until C++20
        //      template< class T >
        //      std::complex<T> conj(const std::complex<T>&z);

        const std::complex<float> z(2.3f, 2.1f);
        const auto complex_val_res = dpl::conj(z);

        std::cout << "real : " << complex_val_res.real() << std::endl;
        std::cout << "imag : " << complex_val_res.imag() << std::endl;

        is_done = 1;
    }
#endif

    return TestUtils::done(is_done);
}
