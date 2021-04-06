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
    ZPOSV computes the solution to a complex system of linear equations
       A * X = B,
    where A is an N-by-N Hermitian positive definite matrix and X and B
    are N-by-NRHS matrices.
    The Cholesky decomposition is used to factor A as
       A = U**H * U,  if UPLO = MagmaUpper, or
       A = L * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and  L is a lower triangular
    matrix.  The factored form of A is then used to solve the system of
    equations A * X = B.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H*U or A = L*L**H.

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

    @ingroup magma_posv
*******************************************************************************/
extern "C" magma_int_t
magma_zposv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info )
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
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        *info = -1;
    if ( n < 0 )
        *info = -2;
    if ( nrhs < 0)
        *info = -3;
    if ( lda < max(1, n) )
        *info = -5;
    if ( ldb < max(1, n) )
        *info = -7;
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
    magma_zpotrf_gpu( uplo, n, dA(0,0), ldda, info );
    if ( *info == MAGMA_ERR_DEVICE_ALLOC ) {
        magma_queue_destroy( queue );
        magma_free( dA );
        magma_free( dB );
        goto CPU_INTERFACE;
    }
    magma_zgetmatrix( n, n, dA(0,0), ldda, A, lda, queue );
    if ( *info == 0 ) {
        magma_zsetmatrix( n, nrhs, B, ldb, dB(0,0), lddb, queue );
        magma_zpotrs_gpu( uplo, n, nrhs, dA(0,0), ldda, dB(0,0), lddb, info );
        magma_zgetmatrix( n, nrhs, dB(0,0), lddb, B, ldb, queue );
    }
    magma_queue_destroy( queue );
    magma_free( dA );
    magma_free( dB );
    return *info;

CPU_INTERFACE:
    /* If multi-GPU or allocation failed, use CPU interface and LAPACK.
     * Faster to use LAPACK for potrs than to copy A to GPU. */
    magma_zpotrf( uplo, n, A, lda, info );
    if ( *info == 0 ) {
        lapackf77_zpotrs( lapack_uplo_const(uplo), &n, &nrhs, A, &lda, B, &ldb, info );
    }
    return *info;
}
