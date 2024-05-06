/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/
#ifndef MAGMASPARSE_ATOMICOPS_MAGMAFLOATCOMPLEX_H
#define MAGMASPARSE_ATOMICOPS_MAGMAFLOATCOMPLEX_H

#include "magmasparse_internal.h"

extern __inline__ __device__ void 
atomicAddmagmaFloatComplex(magmaFloatComplex *addr, magmaFloatComplex val)
{
  dpct::atomic_fetch_add<float, sycl::access::address_space::generic_space>(
      (reinterpret_cast<float(*)[2]>(&addr[0]))[0], MAGMA_Z_REAL(val));
  dpct::atomic_fetch_add<float, sycl::access::address_space::generic_space>(
      (reinterpret_cast<float(*)[2]>(&addr[0]))[1], MAGMA_Z_IMAG(val));
}

#endif 
