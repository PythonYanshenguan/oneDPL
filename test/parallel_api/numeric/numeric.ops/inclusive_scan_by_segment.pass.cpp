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
#include "support/utils.h"
#include "support/scan_serial_impl.h"


using namespace TestUtils;

// This macro may be used to analyze source data and test results in test_inclusive_scan_by_segment
// WARNING: in the case of using this macro debug output is very large.
//#define DUMP_CHECK_RESULTS

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

template <typename BinaryOperation>
struct test_inclusive_scan_by_segment
{
    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, ... };

        Size segment_length = 1;
        for (Size i = 0; i != n; )
        {
          for (Size j = 0; j != 4*segment_length && i != n; ++j)
          {
              host_keys[i] = j/segment_length + 1;
              host_vals[i] = 1;
              host_val_res[i] = 0;
              ++i;
          }
          ++segment_length;
        }
    }

//#ifdef DUMP_CHECK_RESULTS
//#endif // DUMP_CHECK_RESULTS

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void
    check_values(const char* msg, Iterator1 host_keys, Iterator2 host_vals, Iterator3 val_res, Size n,
                 BinaryOperationCheck op = BinaryOperationCheck())
    {
        // https://docs.oneapi.io/versions/latest/onedpl/extension_api.html
        // keys:   [ 0, 0, 0, 1, 1, 1 ]
        // values: [ 1, 2, 3, 4, 5, 6 ]
        // result: [ 1, 1 + 2 = 3, 1 + 2 + 3 = 6, 4, 4 + 5 = 9, 4 + 5 + 6 = 15 ]

#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << n << ") : " << std::endl;
        display_param("keys:   ", host_keys, n);
        display_param("values: ", host_vals, n);
        display_param("result: ", val_res,   n);
#endif // DUMP_CHECK_RESULTS

        if (n < 1)
            return;

        using ValT = typename ::std::decay<decltype(val_res[0])>::type;

        std::vector<ValT> expected_val_res(n);
        inclusive_scan_by_segment_serial(host_keys, host_vals, expected_val_res, n, op);

#ifdef DUMP_CHECK_RESULTS
        display_param("exp.res.: ", expected_val_res.data(), n);
#endif // DUMP_CHECK_RESULTS

        for (Size i = 0; i < n; ++i)
        {
            EXPECT_TRUE(val_res[i] == expected_val_res[i], "wrong effect from exclusive_scan_by_segment");
        }
    }

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           BinaryOperation());
        check_values("6", keys_first, vals_first, val_res_first, n, BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
                              void>::type
    operator()(Policy&& /*exec*/, Iterator1 /*keys_first*/, Iterator1 /*keys_last*/, Iterator2 /*vals_first*/, Iterator2 /*vals_last*/,
               Iterator3 /*val_res_first*/, Iterator3 /*val_res_last*/, Size /*n*/)
    {
    }
};

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

    {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "---------- test_algo_three_sequences -----------" << std::endl;
        test_algo_three_sequences<ValueType, test_inclusive_scan_by_segment<BinaryOperation>>();
    }

    return TestUtils::done();
}
