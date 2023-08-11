/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/
#ifndef MAGMASPARSE_ATOMICOPS_FLOAT_H
#define MAGMASPARSE_ATOMICOPS_FLOAT_H

#include "magmasparse_internal.h"

extern __inline__ void 
atomicAddfloat(float *addr, float val)
{
    dpct::atomic_fetch_add<float, sycl::access::address_space::generic_space>(addr, val);
}


#endif 
