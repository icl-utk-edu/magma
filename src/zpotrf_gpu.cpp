/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
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

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative:  Factorize dA using GPU only mode (only uplo=MagmaLower is available);
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_LL_expert_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info, magma_mode_t mode )
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
    magma_int_t j, jb, nb, recnb;
    magmaDoubleComplex *work;
    magma_int_t *dinfo;

    *info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) {
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
    recnb = 128;
    
    if (mode == MagmaHybrid) {
        if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, nb*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
    }
    else {
        if (MAGMA_SUCCESS != magma_imalloc( &dinfo, 1 ) ) {
            /* alloc failed for workspace */
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
    }
    
    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);
    if (mode == MagmaNative)
        magma_setvector( 1, sizeof(magma_int_t), info, 1, dinfo, 1, queues[0]);
    
    if (uplo == MagmaUpper) {
        //=========================================================
        /* Compute the Cholesky factorization A = U'*U. */
        for (j=0; j < n; j += nb) {
            // apply all previous updates to diagonal block,
            // then transfer it to CPU
            jb = min( nb, n-j );
            magma_zherk( MagmaUpper, MagmaConjTrans, jb, j,
                         d_neg_one, dA(0, j), ldda,
                         d_one,     dA(j, j), ldda, queues[1] );
            
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[1] );
                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
            }
            else {
                //Azzam: need to add events to allow overlap
                magma_zpotrf_rectile_native(MagmaUpper, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[1] );
            }

            
            // apply all previous updates to block row right of diagonal block
            if (j+jb < n) {
                magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                             jb, n-j-jb, j,
                             c_neg_one, dA(0, j   ), ldda,
                                        dA(0, j+jb), ldda,
                             c_one,     dA(j, j+jb), ldda, queues[1] );
            }
            
            // simultaneous with above zgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            if (mode == MagmaHybrid) {
                magma_queue_sync( queues[0] );
                lapackf77_zpotrf( MagmaUpperStr, &jb, work, &jb, info );
                magma_zsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
            }
            
            // apply diagonal block to block row right of diagonal block
            if (j+jb < n) {
                magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                             jb, n-j-jb,
                             c_one, dA(j, j),    ldda,
                                    dA(j, j+jb), ldda, queues[1] );
            }
        }
    }
    else {
        //=========================================================
        // Compute the Cholesky factorization A = L*L'.
        for (j=0; j < n; j += nb) {
            // apply all previous updates to diagonal block,
            // then transfer it to CPU
            jb = min( nb, n-j );
            magma_zherk( MagmaLower, MagmaNoTrans, jb, j,
                         d_neg_one, dA(j, 0), ldda,
                         d_one,     dA(j, j), ldda, queues[0] );
            // Azzam: this section of "ifthenelse" can be moved down to the factorize section and I don't think performane wil be affected.
            if (mode == MagmaHybrid) {
                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
            }
            else {
                magma_zpotrf_rectile_native(MagmaLower, jb, recnb,
                                            dA(j, j), ldda, j,
                                            dinfo, info, queues[0] );
            }
            
            // apply all previous updates to block column below diagonal block
            if (j+jb < n) {
                magma_queue_wait_event(queues[1], events[0]);
                magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                             n-j-jb, jb, j,
                             c_neg_one, dA(j+jb, 0), ldda,
                                        dA(j,    0), ldda,
                             c_one,     dA(j+jb, j), ldda, queues[1] );
                magma_event_record(events[1], queues[1]);
            }
            
            // simultaneous with above zgemm, transfer diagonal block,
            // factor it on CPU, and test for positive definiteness
            // Azzam: The above section can be moved here the code will look cleaner.
            if (mode == MagmaHybrid) {
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
            
            // apply diagonal block to block column below diagonal
            if (j+jb < n) {
                magma_queue_wait_event(queues[0], events[1]);
                magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                             n-j-jb, jb,
                             c_one, dA(j,    j), ldda,
                                    dA(j+jb, j), ldda, queues[0] );
                magma_event_record(events[0], queues[0]);
            }
        }
    }
    if (mode == MagmaNative)
        magma_getvector( 1, sizeof(magma_int_t), dinfo, 1, info, 1, queues[0]);

cleanup:
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    if (mode == MagmaHybrid) {
        magma_free_pinned( work );
    }
    else {
        magma_free( dinfo );
    }
    
    return *info;
} /* magma_zpotrf_LL_expert_gpu */

/***************************************************************************//**
    magma_zpotrf_LL_expert_gpu with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_zpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaHybrid;
    magma_zpotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}

/***************************************************************************//**
    magma_zpotrf_LL_expert_gpu with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_zpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaNative;
    magma_zpotrf_LL_expert_gpu(uplo, n, dA, ldda, info, mode);
    return *info;
}
