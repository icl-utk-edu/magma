/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @precisions normal z -> s d c

*/
#include "magma_internal.h"


/***************************************************************************//**
    Purpose
    -------
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R
    using the non-blocking Householder QR.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    dtau    COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param
    dwork   (workspace) DOUBLE PRECISION array, dimension (N)

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v**H

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqr2
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dwork,
    magma_queue_t queue,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*(ldda))
    
    magma_int_t i, k;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(m,n);

    /* Workspace for diagonal entries - restored at the end */
    magmaDoubleComplex *Aks;
    magma_zmalloc( &Aks, k );
    if ( Aks == NULL ) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(*info) );
    }
    else {
        for (i = 0; i < k; ++i) {
            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
            magma_zlarfg_gpu( m-i, dA(i, i), dA(min(i+1,m), i), dtau+i, dwork, &Aks[i], queue );

            if (n-i-1 > 0) {
                /* Apply H(i)' to A(i:m,i+1:n) from the left */
                magma_zlarf_gpu( m-i, n-i-1, dA(i, i), dtau+i, dA(i, i+1), ldda, queue );
            }
        }

        magma_zcopymatrix( 1, k, Aks, 1, dA(0, 0), ldda+1, queue );
    }
    
    magma_free(Aks);
    return *info;
} /* magma_zgeqr2 */
