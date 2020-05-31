// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _PSTL_NUMERIC_FWD_H
#define _PSTL_NUMERIC_FWD_H

#include <type_traits>
#include <utility>

namespace dpstd
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions, according to draft N4659)
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp __brick_transform_reduce(_ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp, _BinaryOperation1,
                             _BinaryOperation2,
                             /*__is_vector=*/std::true_type) noexcept;

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp __brick_transform_reduce(_ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp, _BinaryOperation1,
                             _BinaryOperation2,
                             /*__is_vector=*/std::false_type) noexcept;

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1,
          class _BinaryOperation2, class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&&, _ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp,
                           _BinaryOperation1, _BinaryOperation2, _IsVector,
                           /*is_parallel=*/std::false_type) noexcept;

#if _PSTL_USE_PAR_POLICIES
template <class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2, class _Tp,
          class _BinaryOperation1, class _BinaryOperation2, class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&&, _RandomAccessIterator1, _RandomAccessIterator1, _RandomAccessIterator2,
                           _Tp, _BinaryOperation1, _BinaryOperation2, _IsVector __is_vector,
                           /*is_parallel=*/std::true_type);
#endif

//------------------------------------------------------------------------
// transform_reduce (version with unary and binary functions)
//------------------------------------------------------------------------

template <class _ForwardIterator, class _Tp, class _UnaryOperation, class _BinaryOperation>
_Tp __brick_transform_reduce(_ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation, _UnaryOperation,
                             /*is_vector=*/std::true_type) noexcept;

template <class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
_Tp __brick_transform_reduce(_ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation, _UnaryOperation,
                             /*is_vector=*/std::false_type) noexcept;

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation,
          class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation,
                           _UnaryOperation, _IsVector,
                           /*is_parallel=*/std::false_type) noexcept;

#if _PSTL_USE_PAR_POLICIES
template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation,
          class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation,
                           _UnaryOperation, _IsVector,
                           /*is_parallel=*/std::true_type);
#endif

//------------------------------------------------------------------------
// transform_exclusive_scan
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
std::pair<_OutputIterator, _Tp> __brick_transform_scan(_ForwardIterator, _ForwardIterator, _OutputIterator,
                                                       _UnaryOperation, _Tp, _BinaryOperation,
                                                       /*Inclusive*/ std::false_type) noexcept;

template <class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
std::pair<_OutputIterator, _Tp> __brick_transform_scan(_ForwardIterator, _ForwardIterator, _OutputIterator,
                                                       _UnaryOperation, _Tp, _BinaryOperation,
                                                       /*Inclusive*/ std::true_type) noexcept;

template <class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp,
          class _BinaryOperation, class _Inclusive, class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_transform_scan(_ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _OutputIterator, _UnaryOperation, _Tp,
                         _BinaryOperation, _Inclusive, _IsVector,
                         /*is_parallel=*/std::false_type) noexcept;

#if _PSTL_USE_PAR_POLICIES
template <class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp,
          class _BinaryOperation, class _Inclusive, class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy_conditional<_ExecutionPolicy, !std::is_floating_point<_Tp>::value,
                                                                 _OutputIterator>
__pattern_transform_scan(_ExecutionPolicy&&, _RandomAccessIterator, _RandomAccessIterator, _OutputIterator,
                         _UnaryOperation, _Tp, _BinaryOperation, _Inclusive, _IsVector, /*is_parallel=*/std::true_type);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp,
          class _BinaryOperation, class _Inclusive, class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy_conditional<_ExecutionPolicy, std::is_floating_point<_Tp>::value,
                                                                 _OutputIterator>
__pattern_transform_scan(_ExecutionPolicy&&, _RandomAccessIterator, _RandomAccessIterator, _OutputIterator,
                         _UnaryOperation, _Tp, _BinaryOperation, _Inclusive, _IsVector, /*is_parallel=*/std::true_type);
#endif

// transform_scan without initial element
template <class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation,
          class _BinaryOperation, class _Inclusive, class _IsVector, class _IsParallel>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                         _OutputIterator __result, _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive,
                         _IsVector __is_vector, _IsParallel __is_parallel);

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator __brick_adjacent_difference(_ForwardIterator, _ForwardIterator, _OutputIterator, _BinaryOperation,
                                            /*is_vector*/ std::false_type) noexcept;

template <class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator __brick_adjacent_difference(_ForwardIterator, _ForwardIterator, _OutputIterator, _BinaryOperation,
                                            /*is_vector*/ std::true_type) noexcept;

template <class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _BinaryOperation,
          class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_adjacent_difference(_ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _OutputIterator, _BinaryOperation,
                              _IsVector, /*is_parallel*/ std::false_type) noexcept;

#if _PSTL_USE_PAR_POLICIES
template <class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _BinaryOperation,
          class _IsVector>
dpstd::__internal::__enable_if_host_execution_policy<_ExecutionPolicy, _OutputIterator>
__pattern_adjacent_difference(_ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _OutputIterator, _BinaryOperation,
                              _IsVector, /*is_parallel*/ std::true_type);
#endif

} // namespace __internal
} // namespace dpstd
#endif /* _PSTL_NUMERIC_FWD_H */
