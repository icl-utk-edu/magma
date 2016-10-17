/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad

   @precisions normal z -> s d c
 */
#include "magma_internal.h"

#define PRECISION_z

#ifdef PRECISION_z
#define POTF2_NB  (8)
#else
#define POTF2_NB  (16)
#endif

#define MAX_NTCOL (1)
#include "zpotf2_devicesfunc.cuh"

#define A(i_, j_)  (dA + (i_) + (j_)*ldda)

/******************************************************************************/
__global__ void zpotf2_smlpin_fixwidth_kernel_native(int m, magmaDoubleComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        if(threadIdx.x < m-i){
            zpotf2_smlpout_fixwidth_device(m-i, A(localstep+i, 0), A(localstep+i, localstep+i), ldda, localstep+i, gbstep, dinfo);
        }
    }
}


/******************************************************************************/
__global__ void zpotf2_smlpin_anywidth_kernel_native(int m, magmaDoubleComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        int ib = min(m-i, POTF2_NB);
        if(threadIdx.x < m-i){
            zpotf2_smlpout_anywidth_device(m-i, ib, A(localstep+i, 0), A(localstep+i, localstep+i), ldda, localstep+i, gbstep, dinfo);
        }
    }
}


/******************************************************************************/
extern "C" magma_int_t
magma_zpotf2_native(
        magma_uplo_t uplo, magma_int_t n, 
        magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t info = 0;
    // Quick return if possible
    if (m == 0 || n == 0) {
        return info;
    }
    dim3 grid(1, 1, 1);
    dim3 threads(m, 1, 1);
    magma_int_t shared_mem_size = sizeof(magmaDoubleComplex) * (m+POTF2_NB)*POTF2_NB;
    if (shared_mem_size > 47000) {
        info = -33;
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    if( m % POTF2_NB == 0){
        zpotf2_smlpin_fixwidth_kernel_native
            <<< grid, threads, shared_mem_size, queue->cuda_stream() >>>
            (m, dA, ldda, 0, gbstep, dinfo);
    }
    else{
        zpotf2_smlpin_anywidth_kernel_native
            <<< grid, threads, shared_mem_size, queue->cuda_stream() >>>
            (m, dA, ldda, 0, gbstep, dinfo);
    }
    return info;
}
