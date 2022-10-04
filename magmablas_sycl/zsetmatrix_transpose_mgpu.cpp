/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Ichitaro Yamazaki
*/
#include "magma_internal.h"

/***************************************************************************//**
    Copy and transpose matrix hA on CPU host to dAT, which is distributed
    row block cyclic over multiple GPUs.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed.
    @param[in]  m       Number of rows    of input matrix hA. m >= 0.
    @param[in]  n       Number of columns of input matrix hA. n >= 0.
    @param[in]  nb      Block size. nb >= 0.
    @param[out] hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[in]  dAT     Array of ngpu pointers, one per GPU, that store the
                        disributed n-by-m matrix A^T on the GPUs, each of dimension (ldda,m).
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU. ngpu*ldda >= n.
    @param[out] dwork   Array of ngpu pointers, one per GPU, that store the
                        workspaces on each GPU, each of dimension (2*lddw*nb).
    @param[in]  lddw    Leading dimension of dwork. lddw >= m.
    @param[in]  queues  2D array of dimension (ngpu,2), with two queues per GPU.

    @ingroup magma_setmatrix_transpose
*******************************************************************************/
extern "C" void
magmablas_zsetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaDoubleComplex *hA,       magma_int_t lda,
    magmaDoubleComplex_ptr    dAT[],    magma_int_t ldda,
    magmaDoubleComplex_ptr    dwork[],  magma_int_t lddw,
    magma_queue_t queues[][2] )
{
#define hA(j)       (hA         + (j)*lda)
#define dwork(d, j) (dwork[(d)] + (j)*nb*lddw)
#define dAT(d, j)   (dAT[(d)]   + (j)*nb)

    magma_int_t nqueues = 2, d, j, j_local, id, ib;
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;
    
    if (lda < m || ngpu*ldda < n || lddw < m) {
        fprintf( stderr, "%s: wrong arguments (%lld < %lld), (%lld*%lld < %lld), or (%lld < %lld).\n",
                 __func__,
                 (long long) lda, (long long) m,
                 (long long) ngpu, (long long) ldda, (long long) n,
                 (long long) lddw, (long long) m );
        return;
    }
    
    /* Move data from CPU to GPU by block columns and transpose it */
    for (j=0; j < n; j += nb) {
        d       = (j/nb)%ngpu;
        j_local = (j/nb)/ngpu;
        id      = j_local%nqueues;
        magma_setdevice(d);
        
        ib = min(n-j, nb);
        magma_zsetmatrix_async( m, ib,
                                hA(j),        lda,
                                dwork(d, id), lddw,
                                queues[d][id] );
        
        magmablas_ztranspose( m, ib, dwork(d,id), lddw, dAT(d,j_local), ldda, queues[d][id] );
    }
}
