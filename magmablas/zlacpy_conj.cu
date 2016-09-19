/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define BLOCK_SIZE 64

/******************************************************************************/
// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old zswap routine. Update to new zlacpy code for 2D matrix?

__global__ void zlacpy_conj_kernel(
    int n,
    magmaDoubleComplex *A1, int lda1,
    magmaDoubleComplex *A2, int lda2 )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_Z_CONJ( A1[offset1] );
    }
}


/******************************************************************************/
extern "C" void 
magmablas_zlacpy_conj(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1, 
    magmaDoubleComplex_ptr dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    dim3 threads( BLOCK_SIZE );
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    zlacpy_conj_kernel<<< blocks, threads, 0, queue->cuda_stream() >>>( n, dA1, lda1, dA2, lda2 );
}
