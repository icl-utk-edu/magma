/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

// === Define what BLAS to use ============================================
    #undef  magma_ztrsm
    #define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use =======================================

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_native_gpu(
    magma_int_t hybrid,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
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
    
    magma_int_t j, jb, nextj, nextjb, nb, recnb;
    magmaDoubleComplex *work;
    magma_int_t *dinfo;

    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_zpotrf_nb( n );

    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    if (MAGMA_SUCCESS != magma_imalloc( &dinfo, 1 ) ) {
        /* alloc failed for workspace */
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_queue_t queues[2];
    magma_event_t trsm_event;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&trsm_event);

    magma_setvector( 1, sizeof(int), info, 1, dinfo, 1, queues[0]);

    /* Use blocked code. */
    if (upper) {
     printf("not supported\n");
    }
    else {
        //=========================================================
        // Compute the Cholesky factorization A = L*L'.
        for (j = 0; j < n; j += nb) {
            jb = min(nb, n-j);
            //===============================================
            //  panel factorization
            //===============================================
            if(hybrid == 1){
                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                // factor it on CPU, and test for positive definiteness
                magma_queue_sync( queues[0] );
                lapackf77_zpotrf( MagmaLowerStr, &jb, work, &jb, info );
                magma_zsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[0] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }
            else{

                magma_queue_sync( queues[0] );
                magma_queue_sync( queues[1] );
                magma_zpotrf_lpout(MagmaLower, jb, dA(j, j), ldda, j, dinfo, queues[0] );
                //magma_zpotf2_native(MagmaLower, jb, dA(j, j), ldda, j, dinfo, queues[1] );
                magma_queue_sync( queues[1] );
                magma_queue_sync( queues[0] );
            }
 
            nextj  = j+jb; 
            if (nextj < n) {
                // apply diagonal block to block column below diagonal
                magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                             n-nextj, jb,
                             c_one, dA(j,    j), ldda,
                                    dA(j+jb, j), ldda, queues[0] );
                magma_event_record(trsm_event, queues[0]);
                // update the trailing matrix (split into two portion, next tile, and remaining
                nextjb = min(nb, n-nextj);

                //magma_queue_wait_event(queues[0], trsm_event);
                magma_zherk( MagmaLower, MagmaNoTrans, nextjb, jb,
                             d_neg_one, dA(j+jb, j), ldda,
                             d_one,     dA(j+jb, j+jb), ldda, queues[0] );


                if (nextj+nextjb < n){
                    // continue the update of the panel lower portion
                    magma_queue_wait_event(queues[1], trsm_event);
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 n-nextj-nextjb, nextjb, jb,
                                 c_neg_one, dA(nextj+nextjb, j),     ldda,
                                            dA(nextj,        j),     ldda,
                                 c_one,     dA(nextj+nextjb, nextj), ldda, queues[1] );


                    // update of the lower portion of the trailing matrix after the next panel column
                    magma_zherk( MagmaLower, MagmaNoTrans, n-nextj-nextjb, jb,
                                 d_neg_one, dA(nextj+nextjb, j),            ldda,
                                 d_one,     dA(nextj+nextjb, nextj+nextjb), ldda, queues[1] );


                }

            }
        }
        
    }
    *info=0;
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( trsm_event );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} /* magma_zpotrf_gpu */
