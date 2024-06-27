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
    Purpose
    -------
    ZGESV_RBT solves a system of linear equations
        A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    Random Butterfly Tranformation is applied on A and B, then
    the LU decomposition with no pivoting is
    used to factor A as
        A = L * U,
    where L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.
    The solution can then be improved using iterative refinement.

    Arguments
    ---------
    @param[in]
    refine  magma_bool_t
            Specifies if iterative refinement is to be applied to improve the solution.
      -     = MagmaTrue:   Iterative refinement is applied.
      -     = MagmaFalse:  Iterative refinement is not applied.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in,out]
    B       COMPLEX_16 array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_gesv_rbt
*******************************************************************************/
extern "C" magma_int_t
magma_zgesv_rbt(
    magma_bool_t refine, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info)
{
    /* Constants */
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    /* Local variables */
    magma_int_t nn = magma_roundup( n, 4 );
    magmaDoubleComplex *hu=NULL, *hv=NULL;
    magmaDoubleComplex_ptr dA=NULL, dB=NULL, dAo=NULL, dBo=NULL, dwork=NULL, dv=NULL;
    magma_int_t iter;
    magma_queue_t queue=NULL;

    /* Function Body */
    *info = 0;
    if ( ! (refine == MagmaTrue) &&
         ! (refine == MagmaFalse) ) {
        *info = -1;
    }
    else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (nrhs == 0 || n == 0)
        return *info;

    // TODO: investigate failures on AMD GPUs
    // For now ignore refine and always set it to False
    // there is probably a bug in the refinement code for the HIP backend
    #ifdef MAGMA_HAVE_HIP
    refine = MagmaFalse
    #endif

    if (MAGMA_SUCCESS != magma_zmalloc( &dA, nn*nn ) ||
        MAGMA_SUCCESS != magma_zmalloc( &dB, nn*nrhs ))
    {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        goto cleanup;
    }

    if (refine == MagmaTrue) {
        if (MAGMA_SUCCESS != magma_zmalloc( &dAo,   nn*nn ) ||
            MAGMA_SUCCESS != magma_zmalloc( &dwork, nn*nrhs ) ||
            MAGMA_SUCCESS != magma_zmalloc( &dBo,   nn*nrhs ))
        {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
    }

    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &hu, 2*nn ) ||
        MAGMA_SUCCESS != magma_zmalloc_cpu( &hv, 2*nn ))
    {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto cleanup;
    }

    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    magmablas_zlaset( MagmaFull, nn, nn, c_zero, c_one, dA, nn, queue );

    /* Send matrix to the GPU */
    magma_zsetmatrix( n, n, A, lda, dA, nn, queue );

    /* Send b to the GPU */
    magma_zsetmatrix( n, nrhs, B, ldb, dB, nn, queue );

    *info = magma_zgerbt_gpu( MagmaTrue, nn, nrhs, dA, nn, dB, nn, hu, hv, info );
    if (*info != MAGMA_SUCCESS)  {
        return *info;
    }

    if (refine == MagmaTrue) {
        magma_zcopymatrix( nn, nn, dA, nn, dAo, nn, queue );
        magma_zcopymatrix( nn, nrhs, dB, nn, dBo, nn, queue );
    }
    /* Solve the system U^TAV.y = U^T.b on the GPU */
    magma_zgesv_nopiv_gpu( nn, nrhs, dA, nn, dB, nn, info );

    /* Iterative refinement */

    if (refine == MagmaTrue) {
        magma_zgerfs_nopiv_gpu( MagmaNoTrans, nn, nrhs, dAo, nn, dBo, nn, dB, nn, dwork, dA, &iter, info );
    }

    /* The solution of A.x = b is Vy computed on the GPU */
    if (MAGMA_SUCCESS != magma_zmalloc( &dv, 2*nn )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        goto cleanup;
    }

    magma_zsetvector( 2*nn, hv, 1, dv, 1, queue );

    magmablas_zprbt_mv(nn, nrhs, dv, dB, nn, queue);

    magma_zgetmatrix( n, nrhs, dB, nn, B, ldb, queue );

cleanup:
    magma_queue_destroy( queue );

    magma_free_cpu( hu );
    magma_free_cpu( hv );

    magma_free( dA );
    magma_free( dv );
    magma_free( dB );

    if (refine == MagmaTrue) {
        magma_free( dAo );
        magma_free( dBo );
        magma_free( dwork );
    }

    return *info;
}
