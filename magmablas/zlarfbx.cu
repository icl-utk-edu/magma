/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"
#include "commonblas_z.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/******************************************************************************/
extern "C"
__global__ void 
magma_zgemv_kernel1(int m, const magmaDoubleComplex * __restrict__ V, int ldv, 
                    const magmaDoubleComplex * __restrict__ c, 
                    magmaDoubleComplex *dwork)
{
    const int i = threadIdx.x;
    const magmaDoubleComplex *dV = V + (blockIdx.x) * ldv;

    __shared__ magmaDoubleComplex sum[ BLOCK_SIZE ];
    magmaDoubleComplex lsum;

    /*  lsum := v**H * C  */
    lsum = MAGMA_Z_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( dV[j] ), c[j] );
    
    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i == 0)
       dwork [blockIdx.x] = sum[0];
}

/******************************************************************************/
/*
    Call 
        magma_zgemv_kernel3<<< n, BLOCK_SIZE, 0, queue->cuda_stream() >>>(m, V, ldv, c, dwork, tau)
    to compute
        ZGEMV( "Conjugate transpose", m, n, -tau[0], V, ldv, c, 1, zero, dwork, 1)
        and to set c[0] to 1.
    i.e., 
        work = -tau[0] V**H c
*/
extern "C"
__global__ void
magma_zgemv_kernel3(int m, const magmaDoubleComplex * __restrict__ V, int ldv, magmaDoubleComplex *c,
                    magmaDoubleComplex *dwork, magmaDoubleComplex *tau)
{
    const int i = threadIdx.x;
    const magmaDoubleComplex *dV = V + (blockIdx.x) * ldv;

    __shared__ magmaDoubleComplex sum[ BLOCK_SIZE ];
    magmaDoubleComplex lsum;

    if (i == 0)
       c[0] = MAGMA_Z_ONE;           

    /*  lsum := v**H * C  */
    lsum = MAGMA_Z_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( dV[j] ), c[j] );

    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i == 0)
       dwork [blockIdx.x] = -tau[0]*sum[0];
}


/******************************************************************************/
extern "C"
__global__ void
magma_zgemv_kernel2(int m, int n, const magmaDoubleComplex * __restrict__ V, int ldv, 
                    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    magmaDoubleComplex lsum;

    V += j;

    lsum = MAGMA_Z_ZERO;
    if (j < m) {
        for (int k=0; k < n; k++)
            lsum += MAGMA_Z_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


/******************************************************************************/
/*
    Apply a complex block reflector H to a complex vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V**H
    where T is the complex k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a complex block of
    k elementary reflectors. 
*/
extern "C" void
magma_zlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue )
{
    /* dwork = V**H c     */
    magma_zgemv_kernel1
        <<< k, BLOCK_SIZE, 0, queue->cuda_stream() >>>
        (m, V, ldv, c, dwork); 

    /* dwork = T**H dwork */
    magma_ztrmv_tkernel
        <<< k, k, 0, queue->cuda_stream() >>>
        ( dT, ldt, dwork, dwork+k);
 
    /* c = c - V dwork    */
    dim3  blocks3( magma_ceildiv( m, BLOCK_SIZE ) );
    dim3 threads3( BLOCK_SIZE );     
    magma_zgemv_kernel2
        <<< blocks3, threads3, 0, queue->cuda_stream() >>>
        ( m, k, V, ldv, dwork+k, c);
}
