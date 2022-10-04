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
    Copy and transpose matrix dAT, which is distributed row block cyclic
    over multiple GPUs, to hA on CPU host.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed.
    @param[in]  m       Number of rows    of output matrix hA. m >= 0.
    @param[in]  n       Number of columns of output matrix hA. n >= 0.
    @param[in]  nb      Block size. nb >= 0.
    @param[in]  dAT     Array of ngpu pointers, one per GPU, that store the
                        disributed n-by-m matrix A^T on the GPUs, each of dimension (ldda,m).
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU. ngpu*ldda >= n.
    @param[out] hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[out] dwork   Array of ngpu pointers, one per GPU, that store the
                        workspaces on each GPU, each of dimension (2*lddw*nb).
    @param[in]  lddw    Leading dimension of dwork. lddw >= m.
    @param[in]  queues  2D array of dimension (ngpu,2), with two queues per GPU.

    @ingroup magma_getmatrix_transpose
*******************************************************************************/
extern "C" void
magmablas_zgetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_const_ptr const dAT[],   magma_int_t ldda,
    magmaDoubleComplex                *hA,      magma_int_t lda,
    magmaDoubleComplex_ptr             dwork[], magma_int_t lddw,
    magma_queue_t queues[][2] )
{
#define hA(j)       (hA         + (j)*lda)
#define dwork(d, j) (dwork[(d)] + (j)*nb*lddw)
#define dAT(d, j)   (dAT[(d)]   + (j)*nb)

    magma_int_t nqueues = 2, d, j, j_local, id, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard argument checking (xerbla)
    if (lda < m || ngpu*ldda < n || lddw < m) {
        fprintf( stderr, "%s: wrong arguments (%lld < %lld), (%lld*%lld < %lld), or (%lld < %lld).\n",
                 __func__,
                 (long long) lda, (long long) m,
                 (long long) ngpu, (long long) ldda, (long long) n,
                 (long long) lddw, (long long) m );
        return;
    }
    
    /* Move data from GPU to CPU using two buffers; first transpose the data on the GPU */
    for (j=0; j < n; j += nb) {
        d       = (j/nb)%ngpu;
        j_local = (j/nb)/ngpu;
        id      = j_local%nqueues;
        magma_setdevice(d);
        
        ib = min(n-j, nb);
        magmablas_ztranspose( ib, m, dAT(d,j_local), ldda, dwork(d,id), lddw, queues[d][id] );
        magma_zgetmatrix_async( m, ib,
                                dwork(d, id), lddw,
                                hA(j),        lda,
                                queues[d][id] );
    }
}
