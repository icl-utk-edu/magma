/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/
#ifndef MAGMASPARSE_ATOMICOPS_MAGMADOUBLECOMPLEX_H
#define MAGMASPARSE_ATOMICOPS_MAGMADOUBLECOMPLEX_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

  // for CUDA_VERSION

#if (defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) ||                        \
    (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP < 600))

__forceinline__ __device__ static double 
atomicAdd(double *addr, double val)
{
    double old = *addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(
                    atomicCAS((unsigned long long int*)addr,
                              __double_as_longlong(assumed),
                              __double_as_longlong(val+assumed)));
    } while(assumed != old);

    return old;
}
#endif

extern __dpct_inline__ void
atomicAddmagmaDoubleComplex(magmaDoubleComplex *addr, magmaDoubleComplex val)
{
    dpct::atomic_fetch_add<double, sycl::access::address_space::generic_space>(
        (reinterpret_cast<double(*)[2]>(&addr[0]))[0], val.real());
//        &(addr[0].x()), val.x());
    dpct::atomic_fetch_add<double, sycl::access::address_space::generic_space>(
        (reinterpret_cast<double(*)[2]>(&addr[0]))[1], val.imag());
        //&(addr[0].y()), val.y());
}


#endif 
