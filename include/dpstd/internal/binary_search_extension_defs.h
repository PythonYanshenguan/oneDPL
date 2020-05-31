/*
 *  Copyright (c) 2020 Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef DPCPP_BINARY_SEARCH_EXTENSION_DEFS_H_
#define DPCPP_BINARY_SEARCH_EXTENSION_DEFS_H_

#include <dpstd/pstl/dpstd_config.h>

namespace dpstd {

    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    lower_bound(Policy&& policy, InputIterator1 start, InputIterator1 end,
		InputIterator2 value_start, InputIterator2 value_end, OutputIterator result);
  
    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    lower_bound(Policy&& policy, InputIterator1 start, InputIterator1 end,
		InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp);
  
    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    upper_bound(Policy&& policy, InputIterator1 start, InputIterator1 end,
		InputIterator2 value_start, InputIterator2 value_end, OutputIterator result);
  
    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    upper_bound(Policy&& policy, InputIterator1 start, InputIterator1 end,
		InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp);

    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    binary_search(Policy&& policy, InputIterator1 start, InputIterator1 end,
                InputIterator2 value_start, InputIterator2 value_end, OutputIterator result);

    template<typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
    dpstd::__internal::__enable_if_execution_policy<Policy, OutputIterator>
    binary_search(Policy&& policy, InputIterator1 start, InputIterator1 end,
                InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp);


} // end namespace dpstd

#endif
