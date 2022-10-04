#ifndef ICL_MAGMA_SYNC_CUH
#define ICL_MAGMA_SYNC_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

/******************************************************************************/
/**                              WARP SYNC                                   **/
/******************************************************************************/
static inline void magmablas_syncwarp(sycl::nd_item<3> item_ct1) 
{
#if __CUDACC_VER_MAJOR__ >= 9
    sycl::group_barrier(item_ct1.get_sub_group());
#else
    // assume implicit warp synchronization
    // using syncthreads() is not safe here
    // as the warp can be part of a bigger thread block
#endif
}


#endif    // ICL_MAGMA_SYNC_CUH

