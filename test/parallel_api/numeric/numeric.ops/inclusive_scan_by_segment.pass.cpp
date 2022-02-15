// -*- C++ -*-
//===-- inclusive_scan_by_segment.pass.cpp ------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/pstl/numeric_fwd.h"

#include "support/test_config.h"


template<typename ViewKeys, typename ViewVals, typename Res, typename Size, typename BinaryOperation>
void inclusive_scan_by_segment_serial(ViewKeys keys, ViewVals vals, Res& res, Size n, BinaryOperation binary_op)
{
    for (Size i = 0; i < n; ++i)
        if (i == 0 || keys[i] != keys[i - 1])
            res[i] = vals[i];
        else
            res[i] = binary_op(res[i - 1], vals[i]);
}

template <typename Iterator, typename Size>
void display_param(const char* msg, Iterator it, Size n)
{
    std::cout << msg;
    for (Size i = 0; i < n; ++i)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << it[i];
    }
    std::cout << std::endl;
}

template<typename _Tp>
struct UserBinaryOperation
{
    _Tp operator()(const _Tp& __x, const _Tp& __y) const
    {
        return __x * __y;
    }
};

int main()
{
    constexpr size_t n = 7;

    using ValueType = ::std::int64_t;
    using BinaryOperation = UserBinaryOperation<ValueType>;

    std::vector<ValueType> key  = { 1, 2, 3, 4, 1, 1, 2 };
    std::vector<ValueType> val  = { 1, 1, 1, 1, 1, 1, 1 };


    using FlagType = unsigned int;
    ::std::tuple<ValueType, FlagType> init(1, 1);
    BinaryOperation binary_op;

    {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "---------- FORWARD ITERATORS -----------" << std::endl;
        display_param("key  : ", key, n);
        display_param("val  : ", val, n);

        std::vector<ValueType> mask = { 1, 1, 1, 1, 1, 0, 1 };    // Mask for forward iterator
        display_param("mask : ", mask.begin(), n);

        std::vector<ValueType> results(n);

        auto first2 = val.begin();
        auto result = results.begin();

        std::cout << std::endl;
        std::cout << "Run oneapi::dpl::__internal::__brick_transform_scan with forward iterators..." << std::endl;

        // Prepare params
        auto __first     = oneapi::dpl::make_zip_iterator(first2, mask.data());
        auto __last      = oneapi::dpl::make_zip_iterator(first2, mask.data()) + n;
        auto __result    = oneapi::dpl::make_zip_iterator(result, mask.data());
        auto __no_op     = oneapi::dpl::__internal::__no_op();
        auto __binary_op = oneapi::dpl::internal::segmented_scan_fun<ValueType, FlagType, BinaryOperation>(binary_op);

        // Call algorithm : __brick_transform_scan -> __unseq_backend::__simd_scan
        oneapi::dpl::__internal::__brick_transform_scan(
            __first, __last, __result, __no_op, init, __binary_op,
            ::std::true_type(),                                     // Inclusive
            ::std::true_type());                                    // is_vector

        // Results:
        display_param("results :", results, n);

        // Expected results:
        std::vector<ValueType> excpected_results(n);
        inclusive_scan_by_segment_serial(key, val, excpected_results, n, binary_op);
        display_param("exp.res.:", excpected_results, n);
    }

    {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "---------- REVERSE ITERATORS -----------" << std::endl;
        display_param("key  : ", key, n);
        display_param("val  : ", val, n);

        std::vector<ValueType> mask = { 1, 0, 1, 1, 1, 1, 1 };    // Mask for reverse iterator
        display_param("mask : ", mask.rbegin(), n);

        std::vector<ValueType> results(n);

        auto first2 = val.rbegin();
        auto result = results.rbegin();

        std::cout << std::endl;
        std::cout << "Run oneapi::dpl::__internal::__brick_transform_scan with reverse iterators..." << std::endl;

        // Prepare params
        auto __first     = oneapi::dpl::make_zip_iterator(first2, mask.data());
        auto __last      = oneapi::dpl::make_zip_iterator(first2, mask.data()) + n;
        auto __result    = oneapi::dpl::make_zip_iterator(result, mask.data());
        auto __no_op     = oneapi::dpl::__internal::__no_op();
        auto __binary_op = oneapi::dpl::internal::segmented_scan_fun<ValueType, FlagType, BinaryOperation>(binary_op);

        // Call algorithm : __brick_transform_scan -> __unseq_backend::__simd_scan
        oneapi::dpl::__internal::__brick_transform_scan(
            __first, __last, __result, __no_op, init, __binary_op,
            ::std::true_type(),                                     // Inclusive
            ::std::true_type());                                    // is_vector

        // Results:
        display_param("results :", results, n);

        // Expected results:
        std::vector<ValueType> excpected_results(n);
        inclusive_scan_by_segment_serial(key, val, excpected_results, n, binary_op);
        display_param("exp.res.:", excpected_results, n);
    }

    return 0;
}
