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

// rocBLAS has a very slow syrk/herk
// switch to magmablas syrk/herk (which internally calls rocBLAS GEMM)
#if defined(MAGMA_HAVE_HIP)
#define magma_zherk    magmablas_zherk
#endif

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
    This algorithm uses left-looking Cholesky factorization

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored (hybrid mode only);
      -     = MagmaLower:  Lower triangle of dA is stored (hybrid & native modes).

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
    ldda    INTEGER
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
      -     = MagmaNative:  Factorize dA using GPU only mode.
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @param[in]
    nb      INTEGER
            The blocking size used during the factorization. nb > 0;
            Users with no specific preference of nb can call magma_get_zpotrf_nb()
            to get the value of nb as determined by MAGMA's internal tuning.

    @param[in]
    recnb   INTEGER
            The blocking size used during the recursive panel factorization (0 < recnb <= nb);
            Users with no specific preference of recnb can set it to a fixed value of
            64 or 128.

    @param[in,out]
    host_work  Workspace, allocated on host (CPU) memory. For faster CPU-GPU communication,
               user can allocate it as pinned memory using magma_malloc_pinned()

    @param[in,out]
    lwork_host   INTEGER pointer
                 The size of the workspace (host_work) in bytes
                 - lwork_host[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork_host. The workspace itself is not referenced, and no
                   factorization is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork_host.

    @param[in,out]
    device_work  Workspace, allocated on device (GPU) memory.

    @param[in,out]
    lwork_device   INTEGER pointer
                   The size of the workspace (device_work) in bytes
                   - lwork_device[0] < 0: a workspace query is assumed, the routine
                     calculates the required amount of workspace and returns
                     it in lwork_device. The workspace itself is not referenced, and no
                     factorization is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[in]
    events        magma_event_t array of size two
                  - created/destroyed by the user outside the routine
                  - Used to manage inter-stream dependencies

    @param[in]
    queues        magma_queue_t array of size two
                  - created/destroyed by the user outside the routine
                  - Used for concurrent kernel execution, if possible

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_expert_gpu_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info,
    magma_mode_t mode,
    magma_int_t nb, magma_int_t recnb,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_event_t events[2], magma_queue_t queues[2] )
{
    #ifdef MAGMA_HAVE_OPENCL
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
    magma_int_t j, jb;
    magmaDoubleComplex *work = NULL;
    magma_int_t *dinfo = NULL;

    /* Quick return if possible */
    *info = 0;
    if (n == 0) return *info;

    // calculate the required workspace in bytes
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;
    if (mode == MagmaHybrid) {
        if ( nb <= 1 || 4*nb >= n ) {
            h_workspace_bytes += n * n * sizeof(magmaDoubleComplex);
        }
        else {
            h_workspace_bytes += nb * nb * sizeof(magmaDoubleComplex);
        }
    }
    else {
        d_workspace_bytes += 1 * sizeof(magma_int_t);
    }

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return 0;
    }

    // check input arguments
    if (uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    else if( mode != MagmaHybrid && mode != MagmaNative) {
        *info = -6;
    }
    else if(nb <= 0) {
        *info = -7;
    }
    else if(recnb <= 0) {
        *info = -8;
    }
    else if( *lwork_host   < h_workspace_bytes ) {
        *info = -10;
    }
    else if( *lwork_device < d_workspace_bytes ) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // assign pointers
    if (mode == MagmaHybrid) {
        work = (magmaDoubleComplex*) host_work;
    }
    else {
        dinfo = (magma_int_t*) device_work;
    }

    if ( mode == MagmaHybrid && (nb <= 1 || 4*nb >= n) ) {
        /* Use CPU only */
        magma_zgetmatrix( n, n, dA(0,0), ldda, work, n, queues[0]);
        lapackf77_zpotrf(lapack_uplo_const(uplo), &n, work, &n, info );
        magma_zsetmatrix( n, n, work, n, dA(0,0), ldda, queues[0]);
        return *info;
    }

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
                magma_event_record(events[0], queues[0]);
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
        magma_getvector_async( 1, sizeof(magma_int_t), dinfo, 1, info, 1, queues[0]);

    return *info;
} /* magma_zpotrf_expert_gpu_work */

/***************************************************************************//**
    wrapper around magma_zpotrf_expert_gpu_work to hide workspace, event,
    and queue management
    @see magma_zpotrf_expert_gpu_work
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_expert_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info,
    magma_int_t nb, magma_mode_t mode )
{

    *info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    else if( nb <= 0 ) {
        *info = -6;
    }
    else if( mode != MagmaHybrid && mode != MagmaNative ) {
        *info = -7;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0)
        return *info;

    magma_int_t recnb = 128;

    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);

    // query workspace
    magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
    magma_zpotrf_expert_gpu_work(
        uplo, n, NULL, ldda, info,
        mode, nb, recnb,
        NULL, lhwork, NULL, ldwork,
        events, queues );

    // alloc workspace
    void *hwork = NULL, *dwork=NULL;
    if( lhwork[0] > 0 ) {
        magma_malloc_pinned( (void**)&hwork, lhwork[0] );
    }

    if( ldwork[0] > 0 ) {
        magma_malloc( (void**)&dwork, ldwork[0] );
    }

    // main call
    magma_zpotrf_expert_gpu_work(
        uplo, n, dA, ldda, info,
        mode, nb, recnb,
        hwork, lhwork, dwork, ldwork,
        events, queues );

    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    // free workspace
    if( hwork != NULL) {
        magma_free_pinned( hwork );
    }

    if( dwork != NULL ) {
        magma_free( dwork );
    }

    return *info;
} /* magma_zpotrf_expert_gpu */

/***************************************************************************//**
    magma_zpotrf_expert_gpu with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_zpotrf_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
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

    magma_mode_t mode = MagmaHybrid;
    magma_int_t nb = magma_get_zpotrf_nb( n );
    magma_zpotrf_expert_gpu(uplo, n, dA, ldda, info, nb, mode);
    return *info;
}

/***************************************************************************//**
    magma_zpotrf_expert_gpu with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_zpotrf_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
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

    magma_mode_t mode = MagmaNative;
    magma_int_t nb = magma_get_zpotrf_nb( n );
    magma_zpotrf_expert_gpu(uplo, n, dA, ldda, info, nb, mode);
    return *info;
}
