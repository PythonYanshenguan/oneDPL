#include <CL/sycl.hpp>
#include <functional>
#include <iostream>

// <functional>

// reference_wrapper

// operator T& () const;

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelConversionPassTest;

class functor1
{
    // int a;
};

template <class T>
bool
test(T& t)
{
    std::reference_wrapper<T> r(t);
    T& r2 = t;
    return (&r2 == &t);
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelConversionPassTest>([=]() {
            functor1 f1;
            ret_access[0] = test(f1);

            int i = 0;
            ret_access[0] &= test(i);
            const int j = 0;
            ret_access[0] &= test(j);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{
    kernel_test();
    return 0;
}