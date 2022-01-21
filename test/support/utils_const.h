// -*- C++ -*-
//===-- utils_const.h -----------------------------------------------------===//
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

#ifndef UTILS_CONST
#define UTILS_CONST

namespace TestUtils
{
#define _SKIP_RETURN_CODE 77

// Test data ranges other than those that start at the beginning of an input.
constexpr int max_n = 100000;
constexpr int inout1_offset = 3;
constexpr int inout2_offset = 5;
constexpr int inout3_offset = 7;

} /* namespace TestUtils */

#endif // UTILS_CONST