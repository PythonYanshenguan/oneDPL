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
#ifndef _UTILS_TEST_BASE_H
#define _UTILS_TEST_BASE_H

#include <memory>

#include "utils_const.h"
#include "utils_sequence.h"
#include "utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

namespace TestUtils
{
////////////////////////////////////////////////////////////////////////////////
/// enum UDTKind - describe test source data kinds
enum class UDTKind
{
    eKeys = 0,  // Keys
    eVals,      // Values
    eRes        // Results
};

template <typename TEnum>
auto
enum_val_to_index(TEnum enumVal)
    -> decltype(static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal))
{
    return static_cast<typename ::std::underlying_type<TEnum>::type>(enumVal);
}

template <typename TestValueType>
struct test_base_data_visitor;

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data - test source data base class
template <typename TestValueType>
struct test_base_data
{
    /// Check that host buffering is required
    /**
     * @return bool - true, if host buffering of test data is required, false - otherwise
     */
    virtual bool host_buffering_required() const = 0;

    /// Get test data
    /**
     * @param UDTKind kind - test data kind
     * @return TestValueType* - pointer to test data.
     *      ATTENTION: return nullptr, if host buffering is required.
     * @see host_buffering_required
     */
    virtual TestValueType* get_data(UDTKind kind) = 0;

    /// Visit all test data
    /**
     * @param test_base_data_visitor<TestValueType>* visitor - pointer to visitor
     */
    virtual void visit(test_base_data_visitor<TestValueType>* visitor) = 0;
};

#if TEST_DPCPP_BACKEND_PRESENT
////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_usm -  test source data for USM shared/device memory
template <typename TestValueType>
struct test_base_data_usm : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = usm_data_transfer_base<TestValueType>;
        using TSourceDataPtr = ::std::unique_ptr<TSourceData>;

        sycl::usm::alloc alloc_type;        // USM alloc type (shared/device)
        TSourceDataPtr   src_data_usm;      // USM data transfer helper

        template<typename _Size>
        Data(sycl::usm::alloc __alloc_type, sycl::queue __q, _Size __sz)
            : alloc_type(__alloc_type)
        {
            // We use this switch/case because we have test_base_data_visitor interface with
            // virtual functions, arguments of which can't be template classes.
            // So, USM allocation type specified in runtime.
            switch (__alloc_type)
            {
            case sycl::usm::alloc::shared:
                src_data_usm.reset(new usm_data_transfer<sycl::usm::alloc::shared, TestValueType>(__q, __sz));
                break;
            case sycl::usm::alloc::device:
                src_data_usm.reset(new usm_data_transfer<sycl::usm::alloc::device, TestValueType>(__q, __sz));
                break;
            default:
                assert(false);
                break;
            }
        }

        /// Get pointer to usm_data_transfer<sycl::usm::alloc::shared, TestValueType> class
        /**
         * Method return pointer from base class to derived class with required spezialization.
         * 
         * @return usm_data_transfer<sycl::usm::alloc::shared, TestValueType>* - pointer
         */
        auto get_usm_data_shared()
        {
            assert(alloc_type == sycl::usm::alloc::shared);
            return reinterpret_cast<usm_data_transfer<sycl::usm::alloc::shared, TestValueType>*>(src_data_usm.get());
        }

        /// Get pointer to usm_data_transfer<sycl::usm::alloc::device, TestValueType> class
        /**
         * Method return pointer from base class to derived class with required spezialization.
         *
         * @return usm_data_transfer<sycl::usm::alloc::device, TestValueType>* - pointer
         */
        auto get_usm_data_device()
        {
            assert(alloc_type == sycl::usm::alloc::device);
            return reinterpret_cast<usm_data_transfer<sycl::usm::alloc::device, TestValueType>*>(src_data_usm.get());
        }

        TestValueType* get_start_from()
        {
            TestValueType* result = nullptr;

            switch (alloc_type)
            {
            case sycl::usm::alloc::shared:
                result = get_usm_data_shared()->get_data();
                break;

            case sycl::usm::alloc::device:
                result = get_usm_data_device()->get_data();
                break;

            default:
                assert(false);
            }

            return result;
        }

        /// Retrieve data from USM shared/device memory
        /**
         * @param _Iterator __it - start iterator
         * @param TDiff __objects_count - retrieving items couunt
         */
        template<typename _Iterator, typename TDiff>
        void retrieve_data(_Iterator __it, TDiff __objects_count)
        {
            switch (alloc_type)
            {
            case sycl::usm::alloc::shared:
                get_usm_data_shared()->retrieve_data(__it, __objects_count);
                break;

            case sycl::usm::alloc::device:
                get_usm_data_device()->retrieve_data(__it, __objects_count);
                break;

            default:
                assert(false);
            }
        }

        /// Update data in USM shared/device memory
        /**
         * @param _Iterator __it - start iterator
         * @param TDiff __objects_count - updating items couunt
         */
        template<typename _Iterator, typename TDiff>
        void update_data(_Iterator __it, TDiff __objects_count)
        {
            switch (alloc_type)
            {
            case sycl::usm::alloc::shared:
                get_usm_data_shared()->update_data(__it, __objects_count);
                break;

            case sycl::usm::alloc::device:
                get_usm_data_device()->update_data(__it, __objects_count);
                break;

            default:
                assert(false);
            }
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 1 item for test1buffer;
                                //  - 2 items for test2buffers;
                                //  - 3 items for test3buffers

    test_base_data_usm(sycl::usm::alloc alloc_type, sycl::queue __q, ::std::initializer_list<::std::size_t> size_list)
    {
        for (auto& size : size_list)
            data.emplace_back(alloc_type, __q, size);
    }

    TestValueType* get_start_from(::std::size_t index)
    {
        auto& data_item = data.at(index);
        return data_item.get_start_from();
    }

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Visit all test data
    virtual void visit(test_base_data_visitor<TestValueType>* visitor) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_buffer - test source data for SYCL buffer
template <typename TestValueType>
struct test_base_data_buffer : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = sycl::buffer<TestValueType, 1>;

        TSourceData   src_data_buf;     // SYCL buffer

        template<typename _Size>
        Data(_Size __sz)
            : src_data_buf(sycl::range<1>(__sz))
        {
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 1 item for test1buffer;
                                //  - 2 items for test2buffers;
                                //  - 3 items for test3buffers

    test_base_data_buffer(::std::initializer_list<Data> init)
        : data(init)
    {
    }

    sycl::buffer<TestValueType, 1>& get_buffer(::std::size_t index)
    {
        return data.at(index).src_data_buf;
    }

    auto get_start_from(::std::size_t index)
        -> decltype(oneapi::dpl::begin(data.at(index).src_data_buf))
    {
        return oneapi::dpl::begin(data.at(index).src_data_buf);
    }

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Visit all test data
    virtual void visit(test_base_data_visitor<TestValueType>* visitor) override;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_sequence -  test source data for sequence (based on std::vector)
template <typename TestValueType>
struct test_base_data_sequence : test_base_data<TestValueType>
{
    struct Data
    {
        using TSourceData = Sequence<TestValueType>;

        TSourceData   src_data_seq;     // Sequence

        Data(::std::size_t size)
            : src_data_seq(size)
        {
        }
    };
    ::std::vector<Data> data;   // Vector of source test data:
                                //  - 3 items for test_algo_three_sequences

    test_base_data_sequence(::std::initializer_list<Data> init)
        : data(init)
    {
    }

    auto get_start_from(::std::size_t index)
        -> decltype(data.at(index).src_data_seq.begin())
    {
        return data.at(index).src_data_seq.begin();
    }

// test_base_data

    // Check that host buffering is required
    virtual bool host_buffering_required() const override;

    // Get test data
    virtual TestValueType* get_data(UDTKind kind) override;

    // Visit all test data
    virtual void visit(test_base_data_visitor<TestValueType>* visitor) override;
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_visitor - interface of source test data visitor
/// By using this interface we may traverse throught all source test data
/// of different kinds: USM shared/device memory, SYCL::buffer and Sequence.
template <typename TestValueType>
struct test_base_data_visitor
{
#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) = 0;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) = 0;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_visitor_impl - base implementation of source test data visitor
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_impl : test_base_data_visitor<TestValueType>
{
    test_base_data_visitor_impl(UDTKind kind, Iterator it_from, Iterator it_to)
        : __kind(kind), __it_from(it_from), __it_to(it_to)
    {
    }

    const UDTKind  __kind;          // Source test data kind (keys, values, results)
    const Iterator __it_from;       // Begin iterator in [begin, end)
    const Iterator __it_to;         // End itedator in [begin, end)
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_visitor_retrieve - implementation of retrieve data visitor
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_retrieve : test_base_data_visitor_impl<TestValueType, Iterator>
{
    using Base = test_base_data_visitor_impl<TestValueType, Iterator>;

    test_base_data_visitor_retrieve(UDTKind kind, Iterator it_from, Iterator it_to)
        : Base(kind, it_from, it_to)
    {
    }

#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) override;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) override;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) override;
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base_data_visitor_update - implementation of update data visitor
template <typename TestValueType, typename Iterator>
struct test_base_data_visitor_update : test_base_data_visitor_impl<TestValueType, Iterator>
{
    using Base = test_base_data_visitor_impl<TestValueType, Iterator>;

    test_base_data_visitor_update(UDTKind kind, Iterator it_from, Iterator it_to)
        : Base(kind, it_from, it_to)
    {
    }

#if TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_usm     <TestValueType>& obj) override;
    virtual bool on_visit(::std::size_t nIndex, test_base_data_buffer  <TestValueType>& obj) override;
#endif // TEST_DPCPP_BACKEND_PRESENT
    virtual bool on_visit(::std::size_t nIndex, test_base_data_sequence<TestValueType>& obj) override;
};

////////////////////////////////////////////////////////////////////////////////
/// struct test_base - base class for new tests
template <typename TestValueType>
struct test_base
{
    test_base_data<TestValueType>& base_data_ref;

    test_base(test_base_data<TestValueType>& _base_data_ref)
        : base_data_ref(_base_data_ref)
    {
    }

    /// Check that host buffering is required
    /**
     * @return bool - true, if host buffering of test data is required, false - otherwise
     */
    bool host_buffering_required() const
    {
        return base_data_ref.host_buffering_required();
    }

    /// class TestDataTransfer - copy test data from/to source test data storage
    /// to/from local buffer for modification processing.
    template <UDTKind kind, typename Size>
    class TestDataTransfer
    {
    public:

        using HostData = std::vector<TestValueType>;
        using Iterator = typename HostData::iterator;

        /// Constructor
        /**
         * @param test_base& _test_base - reference to test base class
         * @param Size _count - count of objects in source test storage
         */
        TestDataTransfer(test_base& _test_base, Size _count)
            : __test_base(_test_base)
            , __host_buffering_required(_test_base.host_buffering_required())
            , __host_buffer(__host_buffering_required ? _count : 0)
            , __count(_count)
        {
        }

        /// Get pointer to internal data buffer
        /**
         * @return TestValueType* - pointer to internal data buffer
         */
        TestValueType* get()
        {
            if (__host_buffering_required)
                return __host_buffer.data();

            return __test_base.base_data_ref.get_data(kind);
        }

        /// Retrieve data
        /**
         * Method copy data from test source data storage (USM shared/device buffer, SYCL buffer)
         * to internal buffer.
         */
        void retrieve_data()
        {
            if (__host_buffering_required)
            {
                test_base_data_visitor_retrieve<TestValueType, Iterator> visitor_retrieve(
                    kind, __host_buffer.begin(), __host_buffer.end());

                __test_base.base_data_ref.visit(&visitor_retrieve);
            }
        }

        /// Update data
        /**
         * Method copy data from internal buffer to test source data storage.
         * 
         * @param Size count - count of items to copy, if 0 - copy all data.
         */
        void update_data(Size count = 0)
        {
            assert(count <= __count);

            if (__host_buffering_required)
            {
                if (count == 0)
                    count = __count;

                test_base_data_visitor_update<TestValueType, Iterator> visitor_update(
                    kind, __host_buffer.begin(), __host_buffer.begin() + count);

                __test_base.base_data_ref.visit(&visitor_update);
            }
        }

    protected:

        test_base& __test_base;     // Test base class ref
        bool       __host_buffering_required = false;
        HostData   __host_buffer;   // Local test data buffer
        const Size __count = 0;     // Count of items in test data
    };
};

namespace
{
void retrieve_data()
{
}

void update_data()
{
}
};

/// Copy data from source test data storage into local buffers
template <typename TTestDataTransfer, typename... Args>
void retrieve_data(TTestDataTransfer& helper, Args&& ...args)
{
    helper.retrieve_data();
    retrieve_data(::std::forward<Args>(args)...);
}

/// Copy data from local buffers into source test data storage
template <typename TTestDataTransfer, typename... Args>
void update_data(TTestDataTransfer& helper, Args&& ...args)
{
    helper.update_data();
    update_data(::std::forward<Args>(args)...);
}

// 1) define class as
//      template <typename TestValueType>
//      struct TestClassName : TestUtils::test_base<TestValueType>
// 2) define class as
//      struct TestClassName
#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST(TestClassName)                                                  \
    template <typename TestValueType>                                               \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST(TestClassName)                                                  \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TestValueType, typename TemplateParams>                      \
    struct TestClassName : TestUtils::test_base<TestValueType>
#else
#define DEFINE_TEST_1(TestClassName, TemplateParams)                                \
    template <typename TemplateParams>                                              \
    struct TestClassName
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)                                                                    \
    TestClassName(test_base_data<TestValueType>& _test_base_data)                                                 \
        : TestUtils::test_base<TestValueType>(_test_base_data)                                                    \
    {                                                                                                             \
    }                                                                                                             \
                                                                                                                  \
    template <UDTKind kind, typename Size>                                                                        \
    using TestDataTransfer = typename TestUtils::test_base<TestValueType>::template TestDataTransfer<kind, Size>; \
                                                                                                                  \
    using UsedValueType = TestValueType;
#else
#define DEFINE_TEST_CONSTRUCTOR(TestClassName)
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData& data)
{
    return TestName(data);
}

template <typename T, typename TestName, typename TestBaseData>
typename ::std::enable_if<!::std::is_base_of<test_base<T>, TestName>::value, TestName>::type
create_test_obj(TestBaseData&)
{
    return TestName();
}

//--------------------------------------------------------------------------------------------------------------------//
// Used with algorithms that have two input sequences and one output sequences
template <typename T, typename TestName>
//typename ::std::enable_if<::std::is_base_of<test_base<T>, TestName>::value, void>::type
void
test_algo_three_sequences()
{
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        using TestBaseData = test_base_data_sequence<T>;

        TestBaseData test_base_data({ max_n, max_n, max_n });

        // create iterators
        auto inout1_first = test_base_data.get_start_from(0);
        auto inout2_first = test_base_data.get_start_from(1);
        auto inout3_first = test_base_data.get_start_from(2);

        invoke_on_all_host_policies()(create_test_obj<T, TestName>(test_base_data),
                                      inout1_first, inout1_first + n,
                                      inout2_first, inout2_first + n,
                                      inout3_first, inout3_first + n,
                                      n);
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestName>
typename ::std::enable_if<
    ::std::is_base_of<test_base<typename TestName::UsedValueType>, TestName>::value,
    void>::type
test_algo_three_sequences()
{
    test_algo_three_sequences<typename TestName::UsedValueType, TestName>();
}

}; // namespace TestUtils

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType>
bool
TestUtils::test_base_data_usm<TestValueType>::host_buffering_required() const
{
    return data.end() != ::std::find_if(data.begin(), data.end(),
                                        [](const Data& item)
                                        {
                                            return item.alloc_type != sycl::usm::alloc::shared;
                                        });
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestValueType*
TestUtils::test_base_data_usm<TestValueType>::get_data(UDTKind kind)
{
    if (host_buffering_required())
        return nullptr;

    return get_start_from(enum_val_to_index(kind));
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_usm<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    bool bProcessed = false;

    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
        {
            bProcessed = true;
            break;
        }
    }

    assert(bProcessed);
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_buffer<TestValueType>::host_buffering_required() const
{
    return true;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestValueType*
TestUtils::test_base_data_buffer<TestValueType>::get_data(UDTKind /*kind*/)
{
    return nullptr;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_buffer<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    bool bProcessed = false;

    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
        {
            bProcessed = true;
            break;
        }
    }

    assert(bProcessed);
}

#endif //  TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
bool
TestUtils::test_base_data_sequence<TestValueType>::host_buffering_required() const
{
    return false;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
TestValueType*
TestUtils::test_base_data_sequence<TestValueType>::get_data(UDTKind kind)
{
    auto& data_item = data.at(enum_val_to_index(kind));
    return data_item.src_data_seq.data();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType>
void
TestUtils::test_base_data_sequence<TestValueType>::visit(test_base_data_visitor<TestValueType>* visitor)
{
    bool bProcessed = false;

    for (::std::size_t nIndex = 0; nIndex < data.size(); ++nIndex)
    {
        if (visitor->on_visit(nIndex, *this))
        {
            bProcessed = true;
            break;
        }
    }

    assert(bProcessed);
}

#if TEST_DPCPP_BACKEND_PRESENT
//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_buffer<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data = obj.data.at(nIndex);
    auto acc = data.src_data_buf.template get_access<sycl::access::mode::read_write>();

    auto __index = 0;
    for (auto __it = Base::__it_from; __it != Base::__it_to; ++__it, ++__index)
    {
        *__it = acc[__index];
    }

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_usm<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data_item = obj.data.at(nIndex);
    data_item.retrieve_data(Base::__it_from, Base::__it_to - Base::__it_from);

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_retrieve<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_sequence<TestValueType>& /*obj*/)
{
    // No additional actions required here

    return nIndex == enum_val_to_index(Base::__kind);
}

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_buffer<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data = obj.data.at(nIndex);

    auto acc = data.src_data_buf.template get_access<sycl::access::mode::read_write>();

    auto __index = 0;
    for (auto __it = Base::__it_from; __it != Base::__it_to; ++__it, ++__index)
    {
        acc[__index] = *__it;
    }

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
#if TEST_DPCPP_BACKEND_PRESENT
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_usm<TestValueType>& obj)
{
    if (nIndex != enum_val_to_index(Base::__kind))
        return false;

    auto& data_item = obj.data.at(nIndex);
    data_item.update_data(Base::__it_from, Base::__it_to - Base::__it_from);

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

//--------------------------------------------------------------------------------------------------------------------//
template <typename TestValueType, typename Iterator>
bool
TestUtils::test_base_data_visitor_update<TestValueType, Iterator>::on_visit(
    ::std::size_t nIndex, TestUtils::test_base_data_sequence<TestValueType>& /*obj*/)
{
    // No additional actions required here

    return nIndex == enum_val_to_index(Base::__kind);
}

//--------------------------------------------------------------------------------------------------------------------//

#endif // _UTILS_TEST_BASE_H
