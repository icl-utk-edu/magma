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

#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#if (defined( CUDA_VERSION ) && ( CUDA_VERSION < 8000 )) \
    || (defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ < 600 ))
    
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

extern __forceinline__ __device__ void 
atomicAddmagmaDoubleComplex(magmaDoubleComplex *addr, magmaDoubleComplex val)
{
    atomicAdd(&(addr[0].x), val.x);
    atomicAdd(&(addr[0].y), val.y);
}


#endif 
