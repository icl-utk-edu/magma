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

// === Define what BLAS to use ============================================
//#undef  magma_ztrsm
//#define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use ======================================

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form
        A = U**H * U,  if uplo = MagmaUpper, or
        A = L  * L**H, if uplo = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    This uses multiple queues to overlap communication and computation.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If uplo = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If uplo = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U or A = L * L**H.
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_expert(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex  *A, magma_int_t lda,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magma_int_t *info, magma_queue_t *queues)
{
    #define  A(i_, j_)  (A + (i_) + (j_)*lda)
    
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif
    
    /* Constants */
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    bool upper = (uplo == MagmaUpper);
    
    magma_int_t j, jb, nb;
    
    /* Check arguments */
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return *info;
    
    nb = magma_get_zpotrf_nb( n );
    
    if (nb <= 1 || 2*nb >= n) {
        lapackf77_zpotrf( uplo_, &n, A, &lda, info );
    }
    else {
        /* Use hybrid blocked code. */
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                /* Update and factorize the current diagonal block and test
                   for non-positive-definiteness. */
                jb = min( nb, n-j );
                magma_zsetmatrix_async( jb, n-j,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[1] );
                
                magma_zherk( MagmaUpper, MagmaConjTrans, jb, j,
                             d_neg_one, dA(0, j), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                magma_queue_sync( queues[1] );
                
                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                         A(j, j), lda, queues[0] );
                
                if (j+jb < n) {
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, n-j-jb, j,
                                 c_neg_one, dA(0, j   ), ldda,
                                            dA(0, j+jb), ldda,
                                 c_one,     dA(j, j+jb), ldda, queues[1] );
                }
                
                magma_queue_sync( queues[0] );
                
                // this could be on any queue; it isn't needed until exit.
                magma_zgetmatrix_async( j, jb,
                                        dA(0, j), ldda,
                                         A(0, j), lda, queues[0] );
                
                lapackf77_zpotrf( MagmaUpperStr, &jb, A(j, j), &lda, info );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                magma_zsetmatrix_async( jb, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[0] );
                magma_queue_sync( queues[0] );
                
                if (j+jb < n) {
                    magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, n-j-jb,
                                 c_one, dA(j, j   ), ldda,
                                        dA(j, j+jb), ldda, queues[1] );
                }
            }
        }
        else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness.
                jb = min( nb, n-j );
                magma_zsetmatrix_async( n-j, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[1] );
                
                magma_zherk( MagmaLower, MagmaNoTrans, jb, j,
                             d_neg_one, dA(j, 0), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                magma_queue_sync( queues[1] );
                
                magma_zgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                         A(j,j), lda, queues[0] );
                
                if (j+jb < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 n-j-jb, jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda,
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda, queues[1] );
                }
                
                magma_queue_sync( queues[0] );
                
                // this could be on any queue; it isn't needed until exit.
                magma_zgetmatrix_async( jb, j,
                                        dA(j, 0), ldda,
                                         A(j, 0), lda, queues[0] );
                
                lapackf77_zpotrf( MagmaLowerStr, &jb, A(j, j), &lda, info );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                magma_zsetmatrix_async( jb, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[0] );
                magma_queue_sync( queues[0] );
                
                if (j+jb < n) {
                    magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j,    j), ldda,
                                        dA(j+jb, j), ldda, queues[1] );
                }
            }
        }
    }

    return *info;
} /* magma_zpotrf */

/******************************************************************************
    magma_zpotrf_expert with two queues.
    @see magma_zpotrf_expert
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info )
{
    magma_int_t nb   = magma_get_zpotrf_nb( n );
    magma_int_t ngpu = magma_num_gpus();
    magma_int_t ldda = magma_roundup( n, 32 );

    magmaDoubleComplex_ptr dA = NULL;

    magma_queue_t queues[2] = { NULL, NULL };

    if ( 2*nb < n && ngpu <= 1) {
        if (MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda )) {
            /* alloc failed so call the non-GPU-resident version */
            return magma_zpotrf_m( ngpu, uplo, n, A, lda, info );
        }

        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[0] );
        magma_queue_create( cdev, &queues[1] );
        
        magma_zpotrf_expert(uplo, n, A, lda, dA, ldda, info, queues );
        
        magma_queue_destroy( queues[0] );
        magma_queue_destroy( queues[1] );
        
        magma_free( dA );
    }
    else {
        /* call multi-GPU non-GPU-resident interface or cleanup code for small sizes */
        return magma_zpotrf_expert( uplo, n, A, lda, dA, ldda, info, queues );
    }
    return *info;
} /* magma_zpotrf */
