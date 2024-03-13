#ifndef ICL_MAGMA_SYNC_CUH
#define ICL_MAGMA_SYNC_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

/******************************************************************************/
/**                              WARP SYNC                                   **/
/******************************************************************************/
static inline void magmablas_syncwarp(sycl::nd_item<3> item_ct1) 
{
    sycl::group_barrier(item_ct1.get_sub_group());
}


#endif    // ICL_MAGMA_SYNC_CUH

