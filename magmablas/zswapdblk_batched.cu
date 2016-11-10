/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"


/******************************************************************************/
/*
 *  Swap diagonal blocks of two matrices.
 *  Each thread block swaps one diagonal block.
 *  Each thread iterates across one row of the block.
 */

__global__ void 
zswapdblk_batched_kernel( int nb, int n_mod_nb,
                  magmaDoubleComplex **dA_array, int ldda, int inca,
                  magmaDoubleComplex **dB_array, int lddb, int incb )
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int batchid = blockIdx.z;
    
    magmaDoubleComplex *dA = dA_array[batchid];
    magmaDoubleComplex *dB = dB_array[batchid];
    
    dA += tx + bx * nb * (ldda + inca);
    dB += tx + bx * nb * (lddb + incb);

    magmaDoubleComplex tmp;
    
    if (bx < gridDim.x-1)
    {
        #pragma unroll
        for( int i = 0; i < nb; i++ ) {
            tmp        = dA[i*ldda];
            dA[i*ldda] = dB[i*lddb];
            dB[i*lddb] = tmp;
        }
    }
    else
    {
        for( int i = 0; i < n_mod_nb; i++ ) {
            tmp        = dA[i*ldda];
            dA[i*ldda] = dB[i*lddb];
            dB[i*lddb] = tmp;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    zswapdblk swaps diagonal blocks of size nb x nb between matrices
    dA and dB on the GPU. It swaps nblocks = ceil(n/nb) blocks.
    For i = 1 .. nblocks, submatrices
    dA( i*nb*inca, i*nb ) and
    dB( i*nb*incb, i*nb ) are swapped.
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns of the matrices dA and dB.  N >= 0.

    @param[in]
    nb      INTEGER
            The size of diagonal blocks.
            NB > 0 and NB <= maximum threads per CUDA block (512 or 1024).

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount). 
             Each is a COMPLEX_16 array dA, dimension (ldda,n)
             The matrix dA.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.
            ldda >= (nblocks - 1)*nb*inca + nb.

    @param[in]
    inca    INTEGER
            The row increment between diagonal blocks of dA. inca >= 0. For example,
            inca = 1 means blocks are stored on the diagonal at dA(i*nb, i*nb),
            inca = 0 means blocks are stored side-by-side    at dA(0,    i*nb).

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array dB, dimension (lddb,n)
             The matrix dB.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dB.
            lddb >= (nblocks - 1)*nb*incb + nb.

    @param[in]
    incb    INTEGER
            The row increment between diagonal blocks of dB. incb >= 0. See inca.
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swapdblk
*******************************************************************************/
extern "C" void 
magmablas_zswapdblk_batched(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nblocks = magma_ceildiv( n, nb );
    magma_int_t n_mod_nb = n % nb;
    
    magma_int_t info = 0;
    if (n < 0) {
        info = -1;
    } else if (nb < 1 || nb > 1024) {
        info = -2;
    } else if (ldda < (nblocks-1)*nb*inca + nb) {
        info = -4;
    } else if (inca < 0) {
        info = -5;
    } else if (lddb < (nblocks-1)*nb*incb + nb) {
        info = -7;
    } else if (incb < 0) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if (n_mod_nb == 0) nblocks += 1; // a dummy thread block for cleanup code
    
    dim3 dimGrid(nblocks, 1, batchCount);
    
    dim3 dimBlock(nb);
    
    if ( nblocks > 0 ) {
        zswapdblk_batched_kernel<<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            ( nb, n_mod_nb, dA_array, ldda, inca,
                  dB_array, lddb, incb );
    }
}
