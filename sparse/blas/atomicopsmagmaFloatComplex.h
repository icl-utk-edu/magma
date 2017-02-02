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

extern __forceinline__ __device__ void 
atomicAddmagmaFloatComplex(magmaFloatComplex *addr, magmaFloatComplex val)
{
    atomicAdd(&(addr[0].x), val.x);
    atomicAdd(&(addr[0].y), val.y);
}

#endif 
