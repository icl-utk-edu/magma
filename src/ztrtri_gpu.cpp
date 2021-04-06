/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Hatem Ltaief
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZTRTRI computes the inverse of a real upper or lower triangular
    matrix dA.

    This is the Level 3 BLAS version of the algorithm.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  A is upper triangular;
      -     = MagmaLower:  A is lower triangular.

    @param[in]
    diag    magma_diag_t
      -     = MagmaNonUnit:  A is non-unit triangular;
      -     = MagmaUnit:     A is unit triangular.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array ON THE GPU, dimension (LDDA,N)
            On entry, the triangular matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of the array dA contains
            the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of the array dA contains
            the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.  If DIAG = MagmaUnit, the
            diagonal elements of A are also not referenced and are
            assumed to be 1.
            On exit, the (triangular) inverse of the original matrix, in
            the same storage format.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value
      -     > 0: if INFO = i, dA(i,i) is exactly zero.  The triangular
                    matrix is singular and its inverse cannot be computed.
                 (Singularity check is currently disabled.)

    @ingroup magma_trtri
*******************************************************************************/
extern "C" magma_int_t
magma_ztrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info)
{
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Constants */
    magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;
    const char* uplo_ = lapack_uplo_const( uplo );
    const char* diag_ = lapack_diag_const( diag );
    
    /* Local variables */
    magma_int_t nb, nn, j, jb;
    magmaDoubleComplex *work;

    bool upper  = (uplo == MagmaUpper);
    bool nounit = (diag == MagmaNonUnit);

    *info = 0;

    if (! upper && uplo != MagmaLower)
        *info = -1;
    else if (! nounit && diag != MagmaUnit)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (ldda < max(1,n))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Check for singularity if non-unit */
    /* cannot do here with matrix dA on GPU -- need kernel */
    /*
    if (nounit) {
        for (j=0; j < n; ++j) {
            if ( MAGMA_Z_EQUAL( *dA(j,j), c_zero )) {
                *info = j+1;  // Fortran index
                return *info;
            }
        }
    }
    */

    /* Determine the block size for this environment */
    nb = magma_get_zpotrf_nb(n);

    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (nb <= 1 || nb >= n) {
        magma_zgetmatrix( n, n, dA(0,0), ldda, work, n, queues[0] );
        lapackf77_ztrtri( uplo_, diag_, &n, work, &n, info );
        magma_zsetmatrix( n, n, work, n, dA(0,0), ldda, queues[0] );
    }
    else if (upper) {
        /* Compute inverse of upper triangular matrix */
        for (j=0; j < n; j += nb) {
            jb = min(nb, n-j);

            if (j > 0) {
                /* Compute rows 0:j of current block column */
                magma_ztrmm( MagmaLeft, MagmaUpper,
                             MagmaNoTrans, diag, j, jb, c_one,
                             dA(0, 0), ldda,
                             dA(0, j), ldda, queues[0] );
    
                magma_ztrsm( MagmaRight, MagmaUpper,
                             MagmaNoTrans, diag, j, jb, c_neg_one,
                             dA(j, j), ldda,
                             dA(0, j), ldda, queues[0] );
            }

            /* Get diagonal block from device */
            magma_zgetmatrix_async( jb, jb,
                                    dA(j, j), ldda,
                                    work,     jb, queues[1] );
            magma_queue_sync( queues[1] );

            /* Compute inverse of current diagonal block */
            lapackf77_ztrtri( MagmaUpperStr, diag_, &jb, work, &jb, info );
            
            /* Send inverted diagonal block to device */
            // use q0, so trsm is done with dA(j,j)
            magma_zsetmatrix_async( jb, jb,
                                    work,     jb,
                                    dA(j, j), ldda, queues[0] );
            magma_queue_sync( queues[0] );  // wait until work is available for next iteration
        }
    }
    else {
        /* Compute inverse of lower triangular matrix */
        nn = ((n-1)/nb)*nb;

        for (j=nn; j >= 0; j -= nb) {
            jb = min(nb, n-j);
            
            if (j+jb < n) {
                /* Compute rows j+jb:n of current block column */
                magma_ztrmm( MagmaLeft, MagmaLower,
                             MagmaNoTrans, diag, n-j-jb, jb, c_one,
                             dA(j+jb, j+jb), ldda,
                             dA(j+jb, j),    ldda, queues[0] );

                magma_ztrsm( MagmaRight, MagmaLower,
                             MagmaNoTrans, diag, n-j-jb, jb, c_neg_one,
                             dA(j, j),    ldda,
                             dA(j+jb, j), ldda, queues[0] );
            }
            
            /* Get diagonal block from device */
            magma_zgetmatrix_async( jb, jb,
                                    dA(j, j), ldda,
                                    work,     jb, queues[1] );
            magma_queue_sync( queues[1] );

            /* Compute inverse of current diagonal block */
            lapackf77_ztrtri( MagmaLowerStr, diag_, &jb, work, &jb, info );
            
            /* Send inverted diagonal block to device */
            // use q0, so trsm is done with dA(j,j)
            magma_zsetmatrix_async( jb, jb,
                                    work,     jb,
                                    dA(j, j), ldda, queues[0] );
            magma_queue_sync( queues[0] );  // wait until work is available for next iteration
        }
    }

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_free_pinned( work );

    return *info;
}
