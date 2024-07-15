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
    ZGESV solves a system of linear equations
       A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as
       A = P * L * U,
    where P is a permutation matrix, L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    ---------
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

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

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

    @ingroup magma_gesv
*******************************************************************************/
extern "C" magma_int_t
magma_zgesv(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info)
{
    #ifdef MAGMA_HAVE_OPENCL
    #define  dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define  dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #else
    #define  dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define  dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #endif
    
    magma_int_t ngpu, ldda, lddb;
    magma_queue_t queue = NULL;
    magma_device_t cdev;
    
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }
    
    /* If single-GPU and allocation suceeds, use GPU interface. */
    ngpu = magma_num_gpus();
    magmaDoubleComplex_ptr dA, dB;
    if ( ngpu > 1 ) {
        goto CPU_INTERFACE;
    }
    ldda = magma_roundup( n, 32 );
    lddb = ldda;
    if ( MAGMA_SUCCESS != magma_zmalloc( &dA, ldda*n )) {
        goto CPU_INTERFACE;
    }
    if ( MAGMA_SUCCESS != magma_zmalloc( &dB, lddb*nrhs )) {
        magma_free( dA );
        goto CPU_INTERFACE;
    }
    
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    magma_zsetmatrix( n, n, A, lda, dA(0,0), ldda, queue );
    magma_zgetrf_gpu( n, n, dA(0,0), ldda, ipiv, info );
    if ( *info == MAGMA_ERR_DEVICE_ALLOC ) {
        magma_queue_destroy( queue );
        magma_free( dA );
        magma_free( dB );
        goto CPU_INTERFACE;
    }
    magma_zgetmatrix( n, n, dA(0,0), ldda, A, lda, queue );
    if ( *info == 0 ) {
        magma_zsetmatrix( n, nrhs, B, ldb, dB(0,0), lddb, queue );
        magma_zgetrs_gpu( MagmaNoTrans, n, nrhs, dA(0,0), ldda, ipiv, dB(0,0), lddb, info );
        magma_zgetmatrix( n, nrhs, dB(0,0), lddb, B, ldb, queue );
    }
    magma_queue_destroy( queue );
    magma_free( dA );
    magma_free( dB );
    return *info;

CPU_INTERFACE:
    /* If multi-GPU or allocation failed, use CPU interface and LAPACK.
     * Faster to use LAPACK for getrs than to copy A to GPU. */
    magma_zgetrf( n, n, A, lda, ipiv, info );
    if ( *info == 0 ) {
        lapackf77_zgetrs( MagmaNoTransStr, &n, &nrhs, A, &lda, ipiv, B, &ldb, info );
    }
    return *info;
}
