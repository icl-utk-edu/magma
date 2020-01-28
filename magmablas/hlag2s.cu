/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"

#define BLK_X 32
#define BLK_Y 4
#define MAX_BATCH    50000

/******************************************************************************/
__device__
void hlag2s_device(
    int m, int n,
    magmaHalf_const_ptr A, int lda,
    float             *SA, int ldsa )
{
#if CUDA_VERSION >= 7500
    const int gtx = blockIdx.x * BLK_X + threadIdx.x;
    const int gty = blockIdx.y * BLK_Y + threadIdx.y;

    for(int j = 0; j < n; j+= gridDim.y) {
        const int gty_ = gty + j;
        for(int i = 0; i < m; i+= gridDim.x) {
            const int gtx_ = gtx + i;
            if(gtx_ < m && gty_ < n) {
                SA[gty_ * ldsa + gtx_] = __half2float( A[gty_ * lda + gtx_] );
            }
        }
    }
#endif
}

/******************************************************************************/
__global__
void hlag2s_kernel(
        int m, int n, 
        magmaHalf_const_ptr dA, int lda, 
        float             *dSA, int ldsa )
{
    hlag2s_device(m, n, dA, lda, dSA, ldsa);
}

/******************************************************************************/
__global__
void hlag2s_kernel_batched(
        int m, int n, 
        magmaHalf const * const * dAarray, int lda, 
        float**                  dSAarray, int ldsa )
{
    const int batchid = blockIdx.z;
    hlag2s_device(m, n, dAarray[batchid], lda, dSAarray[batchid], ldsa);
}

/******************************************************************************/
extern "C" void
magmablas_hlag2s(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA, magma_int_t lda,
    float             *dSA, magma_int_t ldsa,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < max(1,m) )
        info = -4;
    else if ( ldsa < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, BLK_Y );
    dim3 grid(magma_ceildiv( m, BLK_X ), min(50000, magma_ceildiv(n, BLK_Y)), 1);
    hlag2s_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
    ( m, n, dA, lda, dSA, ldsa );
}


/******************************************************************************/
extern "C" void
magmablas_hlag2s_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf const * const * dAarray, magma_int_t lda,
    float               **dSAarray, magma_int_t ldsa,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < max(1,m) )
        info = -4;
    else if ( ldsa < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, BLK_Y );
    const int maxBatch = MAX_BATCH;
    for(int i = 0; i < batchCount; i+=maxBatch){
        magma_int_t batch = min(maxBatch, batchCount-i);
        dim3 grid(magma_ceildiv( m, BLK_X ), magma_ceildiv( n, BLK_Y ), batch);
        hlag2s_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
        ( m, n, dAarray + i, lda, dSAarray + i, ldsa);
    }
}
