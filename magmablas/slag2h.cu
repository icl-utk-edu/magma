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

static __device__ magma_int_t flag = 0;
static __device__ magma_int_t flag_array[ MAX_BATCH ] = { 0 };
/******************************************************************************/
__device__
void slag2h_device(
    int m, int n,
    const float *A, int lda,
    magmaHalf *HA,  int ldha,
    float rmax, magma_int_t* dinfo)
{
#if CUDA_VERSION >= 7500
    const int gtx = blockIdx.x * BLK_X + threadIdx.x;
    const int gty = blockIdx.y * BLK_Y + threadIdx.y;

    float tmp;
    float neg_rmax = - rmax;

    for(int j = 0; j < n; j += gridDim.y) {
        const int gty_ = gty + j;
        for(int i = 0; i < m; i+= gridDim.x){
            const int gtx_ = gtx + i;
            if(gtx_ < m && gty_ < n){
                tmp = A[gty_ * lda + gtx_];
                if ( (MAGMA_S_REAL(tmp) < neg_rmax) || (MAGMA_S_REAL(tmp) > rmax) ) {
                    *dinfo  = 1;
                }
                HA[gty_ * ldha + gtx_] = __float2half( tmp );
            }
        }
    }
#endif
}


/******************************************************************************/
__global__
void slag2h_kernel(
        int m, int n, 
        float const *dA, int lda, 
        magmaHalf* dHA, int ldha, 
        float rmax, magma_int_t* dinfo )
{
    slag2h_device(m, n, dA, lda, dHA, ldha, rmax, dinfo);
}


/******************************************************************************/
__global__
void slag2h_kernel_batched(
        int m, int n, 
        float const * const * dAarray, int lda, 
        magmaHalf** dHAarray, int ldha, 
        float rmax, magma_int_t* dinfo_array, 
        magma_queue_t queue )
{
    const int batchid = blockIdx.z;
    slag2h_device( m, n, dAarray[batchid], lda, dHAarray[batchid], ldha, rmax, &dinfo_array[batchid]);
}

/******************************************************************************/
extern "C" void
magmablas_slag2h(
    magma_int_t m, magma_int_t n,
    float const * dA, magma_int_t lda,
    magmaHalf* dHA, magma_int_t ldha,
    magma_int_t *info, magma_queue_t queue)
{
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldha < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    
    // there is no lapackf77_hlamch, please visit: 
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    dim3 threads( BLK_X, BLK_Y );
    dim3 grid( magma_ceildiv(m, BLK_X), min(50000, magma_ceildiv(n, BLK_Y)), 1);

    slag2h_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
    ( m, n, dA, lda, dHA, ldha, rmax, &flag );

    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag

}


/******************************************************************************/
extern "C" void
magmablas_slag2h_batched(
    magma_int_t m, magma_int_t n,
    float const * const * dAarray, magma_int_t lda,
    magmaHalf** dHAarray, magma_int_t ldha,
    magma_int_t *info_array, magma_int_t batchCount, 
    magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if ( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( lda < max(1,m) )
        arginfo = -4;
    else if ( ldha < max(1,m) )
        arginfo = -6;
    
    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    memset( info_array, 0, batchCount * sizeof(magma_int_t) );    // init info_array to zero

    // there is no lapackf77_hlamch, please visit: 
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    dim3 threads( BLK_X, BLK_Y );
    const int maxBatch = MAX_BATCH;
    for(int i = 0; i < batchCount; i+=maxBatch){
        magma_int_t batch = min(maxBatch, batchCount-i);
        cudaMemcpyToSymbol( flag_array, info_array + i, sizeof(magma_int_t) );

        dim3 grid( magma_ceildiv(m, BLK_X), magma_ceildiv(n, BLK_Y), batch);
        slag2h_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
        ( m, n, dAarray + i, lda, dHAarray + i, ldha, rmax, flag_array, queue);

        cudaMemcpyFromSymbol( info_array + i, flag_array, sizeof(magma_int_t) );
    }
}
