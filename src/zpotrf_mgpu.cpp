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
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    d_lA    COMPLEX_16 array of pointers on the GPU, dimension (ngpu)
            On entry, the Hermitian matrix dA distributed over GPUs
            (d_lA[d] points to the local matrix on the d-th GPU).
            It is distributed in 1D block column or row cyclic (with the
            block size of nb) if UPLO = MagmaUpper or MagmaLower, respectively.
            If UPLO = MagmaUpper, the leading N-by-N upper triangular
            part of dA contains the upper triangular part of the matrix dA,
            and the strictly lower triangular part of dA is not referenced.
            If UPLO = MagmaLower, the leading N-by-N lower triangular part
            of dA contains the lower triangular part of the matrix dA, and
            the strictly upper triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array d_lA. LDDA >= max(1,N).
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
magma_zpotrf_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info)
{
    magma_int_t     j, nb, d, lddp, h;
    const char* uplo_ = lapack_uplo_const( uplo );
    magmaDoubleComplex *work;
    bool upper = (uplo == MagmaUpper);
    magmaDoubleComplex *dwork[MagmaMaxGPUs];
    magma_queue_t    queues[MagmaMaxGPUs][3];
    magma_event_t     event[MagmaMaxGPUs][5];

    *info = 0;
    nb = magma_get_zpotrf_nb(n);
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (!upper) {
        lddp = nb*(n/(nb*ngpu));
        if ( n%(nb*ngpu) != 0 ) lddp += min(nb, n-ngpu*lddp);
        if ( ldda < lddp ) *info = -4;
    } else if ( ldda < n ) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    if (ngpu == 1 && ((nb <= 1) || (nb >= n)) ) {
        /*  Use unblocked code. */
        magma_setdevice(0);
        magma_queue_create( 0, &queues[0][0] );
        if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, n*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_zgetmatrix( n, n, d_lA[0], ldda, work, n, queues[0][0] );
        lapackf77_zpotrf(uplo_, &n, work, &n, info);
        magma_zsetmatrix( n, n, work, n, d_lA[0], ldda, queues[0][0] );
        magma_free_pinned( work );
        magma_queue_destroy( queues[0][0] );
    }
    else {
        lddp = magma_roundup( n, nb );
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            if (MAGMA_SUCCESS != magma_zmalloc( &dwork[d], ngpu*nb*lddp )) {
                for( j=0; j < d; j++ ) {
                    magma_setdevice(j);
                    magma_free( dwork[j] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            for( j=0; j < 3; j++ ) {
                magma_queue_create( d, &queues[d][j] );
            }
            for( j=0; j < 5; j++ ) {
                magma_event_create( &event[d][j]  );
            }
        }
        magma_setdevice(0);
        h = 1; //ngpu; //magma_ceildiv( n, nb );
        if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, n*nb*h )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        if (upper) {
            /* with three queues */
            magma_zpotrf3_mgpu(ngpu, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, n,
                               h, queues, event, info);
        } else {
            /* with three queues */
            magma_zpotrf3_mgpu(ngpu, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, nb*h,
                               h, queues, event, info);
        }

        /* clean up */
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            for( j=0; j < 3; j++ ) {
                magma_queue_sync( queues[d][j] );
                magma_queue_destroy( queues[d][j] );
            }
            
            for( j=0; j < 5; j++ )
                magma_event_destroy( event[d][j] );
            
            magma_free( dwork[d] );
        }
        magma_free_pinned( work );
    } /* end of not lapack */

    magma_setdevice( orig_dev );
    
    return *info;
} /* magma_zpotrf_mgpu */
