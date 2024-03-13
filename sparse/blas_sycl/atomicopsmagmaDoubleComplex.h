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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

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
