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

#ifndef _UTILS_CONST_H
#define _UTILS_CONST_H

namespace TestUtils
{
#define _SKIP_RETURN_CODE 77

// Test data ranges other than those that start at the beginning of an input.
constexpr int max_n = 100000;

} /* namespace TestUtils */

#endif // _UTILS_CONST_H
