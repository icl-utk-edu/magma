/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Copy and transpose matrix hA on CPU host to dAT on GPU device.

    @param[in]  m       Number of rows    of input matrix hA. m >= 0.
    @param[in]  n       Number of columns of input matrix hA. n >= 0.
    @param[in]  nb      Block size. nb >= 0.
    @param[in]  hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[out] dAT     The n-by-m matrix A^T on the GPU, of dimension (ldda,m).
    @param[in]  ldda    Leading dimension of matrix dAT. ldda >= n.
    @param[out] dwork   Workspace on the GPU, of dimension (2*lddw*nb).
    @param[in]  lddw    Leading dimension of dwork. lddw >= m.
    @param[in]  queues  Array of two queues, to pipeline operation.

    @ingroup magma_setmatrix_transpose
*******************************************************************************/
extern "C" void
magmablas_zsetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaDoubleComplex     *hA, magma_int_t lda,
    magmaDoubleComplex_ptr       dAT, magma_int_t ldda,
    magmaDoubleComplex_ptr     dwork, magma_int_t lddw,
    magma_queue_t queues[2] )
{
#define    hA(i_, j_)    (hA + (i_) + (j_)*lda)
#define   dAT(i_, j_)   (dAT + (i_) + (j_)*ldda)
#define dwork(i_, j_) (dwork + (i_) + (j_)*lddw)

    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard check arguments
    if (lda < m || ldda < n || lddw < m) {
        fprintf( stderr, "%s: wrong arguments.\n", __func__ );
        return;
    }

    /* Move data from CPU to GPU in the first panel in the dwork buffer */
    ib = min(n-i, nb);
    magma_zsetmatrix_async( m, ib,
                            hA(0,i), lda,
                            dwork(0,(j%2)*nb), lddw, queues[j%2] );
    j++;

    for (i=nb; i < n; i += nb) {
        /* Move data from CPU to GPU in the second panel in the dwork buffer */
        ib = min(n-i, nb);
        magma_zsetmatrix_async( m, ib,
                                hA(0,i), lda,
                                dwork(0,(j%2)*nb), lddw, queues[j%2] );
        j++;
        
        /* Note that the previous panel (i.e., j%2) comes through the queue
           for the kernel so there is no need to synchronize.             */
        // TODO should this be ib not nb?
        magmablas_ztranspose( m, nb, dwork(0,(j%2)*nb), lddw, dAT(i-nb,0), ldda, queues[j%2] );
    }

    /* Transpose the last part of the matrix.                            */
    j++;
    magmablas_ztranspose( m, ib, dwork(0,(j%2)*nb), lddw, dAT(i-nb,0), ldda, queues[j%2] );
}
