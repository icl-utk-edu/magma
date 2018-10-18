/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "magma_internal.h"

#define dA(i, j)  (dA + (i) + (j)*ldda)
#define PARSWAP
/***************************************************************************//**
    Purpose
    -------
    This is an internal routine.

    ZGETRF_PANEL computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a GPU-only routine. The host CPU is not used. 

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns the matrix A.  N >= 0.

    @param[in,out]
    dA      A COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of A.  LDDA >= max(1,M).

    @param[out]
    dipiv   An INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dipivinfo  An INTEGER array, for internal use.

    @param[out]
    dinfo         INTEGER, stored on the GPU
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[in]
    update_queue   magma_queue_t
                   Internal use.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_recpanel_native(
    magma_int_t m, magma_int_t n,    
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t* dipiv, magma_int_t* dipivinfo,
    magma_int_t *dinfo, magma_int_t gbstep, 
    magma_queue_t queue, magma_queue_t update_queue)
{
    magma_int_t recpnb = 32;
    if (m == 0 || n == 0) {
        return 0;
    }

    magma_int_t panel_nb = n;
    if (panel_nb <= recpnb) {
        magma_zgetf2_native(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue);
        return 0;
    }
    else {
        // split A over two [A A2]
        // panel on A1, update on A2 then panel on A1    
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;

        // panel on A1
        magma_zgetrf_recpanel_native(m, n1, dA(0,0), ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue);

        // update A2
        #ifdef PARSWAP
        setup_pivinfo( dipivinfo, dipiv, m, n1, queue);  // setup pivinfo
        magma_zlaswp_rowparallel_native( n2, dA(0,n1), ldda, dA(0,n1), ldda, 0, n1, dipivinfo, queue );
        #else
        magma_zlaswp_rowserial_native(n2, dA(0,n1), ldda, 0, n1, dipiv, queue);
        #endif

        magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
                     n1, n2, MAGMA_Z_ONE, 
                     dA(0, 0), ldda, 
                     dA(0,n1), ldda, queue );
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                     m-n1, n2, n1, 
                     MAGMA_Z_NEG_ONE, dA(n1,  0), ldda, 
                                      dA(0 , n1), ldda, 
                     MAGMA_Z_ONE,     dA(n1, n1), ldda, queue );

        // panel on A2
        magma_zgetrf_recpanel_native(m-n1, n2, dA(n1,n1), ldda, dipiv+n1, dipivinfo+n1, dinfo, n1, queue, update_queue);

        // swap on the right
        #ifdef PARSWAP
        setup_pivinfo( dipivinfo, dipiv+n1, m-n1, n2, queue);  // setup pivinfo
        #endif

        adjust_ipiv( dipiv+n1, n2, n1, queue);

        #ifdef PARSWAP
        magma_zlaswp_rowparallel_native(n1, dA(n1,0), ldda, dA(n1,0), ldda, n1, n, dipivinfo, queue);
        #else
        magma_zlaswp_rowserial_native(n1, dA(0,0), ldda, n1, n, dipiv, queue);
        #endif
    }
    return 0;
}
