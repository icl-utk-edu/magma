#ifndef ICL_MAGMA_SYNC_CUH
#define ICL_MAGMA_SYNC_CUH

#include "magma_internal.h"

/******************************************************************************/
/**                              WARP SYNC                                   **/
/******************************************************************************/
__device__ static inline void magmablas_syncwarp() 
{
#if __CUDACC_VER_MAJOR__ >= 9
    __syncwarp();
#else
    // assume implicit warp synchronization
    // using syncthreads() is not safe here
    // as the warp can be part of a bigger thread block
#endif
}


#endif    // ICL_MAGMA_SYNC_CUH

