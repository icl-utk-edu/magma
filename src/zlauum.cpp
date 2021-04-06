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
    ZLAUUM computes the product U * U^H or L^H * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array A.

    If UPLO = MagmaUpper then the upper triangle of the result is stored,
    overwriting the factor U in A.
    If UPLO = MagmaLower then the lower triangle of the result is stored,
    overwriting the factor L in A.
    This is the blocked form of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the triangular factor stored in the array A
            is upper or lower triangular:
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the triangular factor U or L.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = MagmaUpper, the upper triangle of A is
            overwritten with the upper triangle of the product U * U^H;
            if UPLO = MagmaLower, the lower triangle of A is overwritten with
            the lower triangle of the product L^H * L.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value

    @ingroup magma_lauum
*******************************************************************************/
extern "C" magma_int_t
magma_zlauum(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info)
{
    #define  A(i_, j_) ( A + (i_) + (j_)*lda )
    
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Constants */
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const double             d_one = MAGMA_D_ONE;
    const char* uplo_ = lapack_uplo_const( uplo );
    
    /* Local variables */
    magma_int_t i, ib, ldda, nb;
    magmaDoubleComplex_ptr dA;
    bool upper = (uplo == MagmaUpper);

    *info = 0;
    if (! upper && uplo != MagmaLower)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,n))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if (n == 0)
        return *info;

    nb = magma_get_zpotrf_nb( n );
    ldda = magma_roundup( n, 32 );

    if (MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (nb <= 1 || nb >= n) {
        lapackf77_zlauum( uplo_, &n, A, &lda, info );
    }
    else if (upper) {
        /* Compute the product U * U^H. */
        // Computing 2nd block column (diagonal & above):
        // [ u11  u12  u13 ]   [ u11^H               ]   [ ...  u12*u22^H + u13*u23^H  ... ]  
        // [      u22  u23 ] * [ u12^H  u22^H        ] = [ ...  u22*u22^H + u23*u23^H  ... ]
        // [           u33 ]   [ u13^H  u23^H  u33^H ]   [ ...  ...                    ... ]
        for (i=0; i < n; i += nb) {
            ib = min( nb, n-i );

            // Send diagonl block, u22
            // This must finish before lauum below
            magma_zsetmatrix( ib, ib,
                              A(i,i),  lda,
                              dA(i,i), ldda, queues[0] );

            // Send right of diagonl block, u23
            magma_zsetmatrix_async( ib, n-i-ib,
                                    A(i,i+ib),  lda,
                                    dA(i,i+ib), ldda, queues[1] );

            // u12 = u12 * u22^H
            magma_ztrmm( MagmaRight, MagmaUpper,
                         MagmaConjTrans, MagmaNonUnit, i, ib, c_one,
                         dA(i,i), ldda,
                         dA(0,i), ldda, queues[0] );

            // u22 = u22 * u22^H
            lapackf77_zlauum( MagmaUpperStr, &ib, A(i,i), &lda, info );
            
            magma_zsetmatrix_async( ib, ib,
                                    A(i,i),  lda,
                                    dA(i,i), ldda, queues[0] );
            
            if (i+ib < n) {
                // wait for u23
                magma_queue_sync( queues[1] );
                
                // u12 += u13 * u23^H
                magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                             i, ib, n-i-ib,
                             c_one, dA(0,i+ib), ldda,
                                    dA(i,i+ib), ldda,
                             c_one, dA(0,i),    ldda, queues[0] );
                
                // u22 += u23 * u23^H
                magma_zherk( MagmaUpper, MagmaNoTrans, ib, n-i-ib,
                             d_one, dA(i,i+ib), ldda,
                             d_one, dA(i,i),    ldda, queues[0] );
            }

            // Get diagonal block & above of current column from device
            // This could be on a different queue -- not needed until return
            magma_zgetmatrix_async( i+ib, ib,
                                    dA(0,i), ldda,
                                    A(0,i),  lda, queues[0] );
        }
    }
    else {
        /* Compute the product L^H * L. */
        for (i=0; i < n; i += nb) {
            ib = min( nb, n-i );
            magma_zsetmatrix( ib, ib,
                              A(i,i),  lda,
                              dA(i,i), ldda, queues[0] );

            magma_zsetmatrix_async( n-i-ib, ib,
                                    A(i+ib,i),  lda,
                                    dA(i+ib,i), ldda, queues[1] );

            magma_ztrmm( MagmaLeft, MagmaLower,
                         MagmaConjTrans, MagmaNonUnit, ib, i, c_one,
                         dA(i,i), ldda,
                         dA(i,0), ldda, queues[0] );


            lapackf77_zlauum( MagmaLowerStr, &ib, A(i,i), &lda, info );

            magma_zsetmatrix_async( ib, ib,
                                    A(i,i),  lda,
                                    dA(i,i), ldda, queues[0] );

            if (i+ib < n) {
                magma_queue_sync( queues[1] );
                
                magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                             ib, i, n-i-ib,
                             c_one, dA(i+ib,i), ldda,
                                    dA(i+ib,0), ldda,
                             c_one, dA(i,0),    ldda, queues[0] );

                magma_zherk( MagmaLower, MagmaConjTrans, ib, n-i-ib,
                             d_one, dA(i+ib,i), ldda,
                             d_one, dA(i,i),    ldda, queues[0] );
            }
            
            magma_zgetmatrix_async( ib, i+ib,
                                    dA(i,0), ldda,
                                    A(i,0),  lda, queues[0] );
        }
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    magma_free( dA );

    return *info;
}
