/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/
#include "magma_internal.h"
#include "magma_timer.h"

/***************************************************************************//**
    Purpose
    -------
    DLAEX0 computes all eigenvalues and the choosen eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    n       INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.
            
    @param[in,out]
    d       DOUBLE PRECISION array, dimension (N)
            On entry, the main diagonal of the tridiagonal matrix.
            On exit, its eigenvalues.
            
    @param[in]
    e       DOUBLE PRECISION array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.
            
    @param[in,out]
    Q       DOUBLE PRECISION array, dimension (LDQ, N)
            On entry, Q will be the identity matrix.
            On exit, Q contains the eigenvectors of the
            tridiagonal matrix.
            
    @param[in]
    ldq     INTEGER
            The leading dimension of the array Q.  If eigenvectors are
            desired, then  LDQ >= max(1,N).  In any case,  LDQ >= 1.
            
    @param
    work    (workspace) DOUBLE PRECISION array,
            the dimension of WORK >= 4*N + N**2.
            
    @param
    iwork   (workspace) INTEGER array,
            the dimension of IWORK >= 3 + 5*N.
            
    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.
            
    @param[in]
    vl      DOUBLE PRECISION
    @param[in]
    vu      DOUBLE PRECISION
            If RANGE=MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.
            
    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            If RANGE=MagmaRangeI, the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeV.
            
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    Further Details
    ---------------
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    @ingroup magma_laex0
*******************************************************************************/
extern "C" magma_int_t
magma_dlaex0_m(
    magma_int_t ngpu,
    magma_int_t n, double *d, double *e,
    double *Q, magma_int_t ldq,
    double *work, magma_int_t *iwork,
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info)
{
#define Q(i_,j_) (Q + (i_) + (j_)*ldq)

    // Constants
    const magma_int_t ione = 1;
    
    // Local variables
    magma_range_t range2;
    magma_int_t curlvl, i, indxq;
    magma_int_t dev, j, k, matsiz, msd2, smlsiz;
    magma_int_t submat, subpbs, tlvls;
    magmaDouble_ptr dw[MagmaMaxGPUs] = { NULL };
    magma_queue_t queues[MagmaMaxGPUs][2] = {{ NULL }};

    // Test the input parameters.
    *info = 0;
    if ( n < 0 )
        *info = -1;
    else if ( ldq < max(1, n) )
        *info = -5;
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0)
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    // workspace dimension for ngpu > 1
    size_t tmp = magma_ceildiv( n, 2 );
    if (ngpu > 1) {
        size_t tmp2 = (tmp-1) / (ngpu/2) + 1;
        tmp = tmp * tmp2 + 2 * magma_get_dlaex3_m_nb()*(tmp + tmp2);
    }

    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        if (ngpu == 1) {
            if (MAGMA_SUCCESS != magma_dmalloc( &dw[dev], 3*n*(n/2 + 1) )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
        }
        else {
            if (MAGMA_SUCCESS != magma_dmalloc( &dw[dev], tmp )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
        }
        magma_queue_create( dev, &queues[dev][0] );
        magma_queue_create( dev, &queues[dev][1] );
    }

    smlsiz = magma_get_smlsize_divideconquer();

    // Determine the size and placement of the submatrices, and save in
    // the leading elements of IWORK.

    iwork[0] = n;
    subpbs= 1;
    tlvls = 0;
    while (iwork[subpbs - 1] > smlsiz) {
        for (j = subpbs; j > 0; --j) {
            iwork[2*j - 1] = (iwork[j-1]+1)/2;
            iwork[2*j - 2] = iwork[j-1]/2;
        }
        ++tlvls;
        subpbs *= 2;
    }
    for (j=1; j < subpbs; ++j)
        iwork[j] += iwork[j-1];

    // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
    // using rank-1 modifications (cuts).

    for (i=0; i < subpbs-1; ++i) {
        submat = iwork[i];
        d[submat-1] -= MAGMA_D_ABS( e[submat-1] );
        d[submat]   -= MAGMA_D_ABS( e[submat-1] );
    }

    indxq = 4*n + 3;

    // Solve each submatrix eigenproblem at the bottom of the divide and
    // conquer tree.
    magma_timer_t time;  time=0;
    timer_start( time );

    for (i = 0; i < subpbs; ++i) {
        if (i == 0) {
            submat = 0;
            matsiz = iwork[0];
        } else {
            submat = iwork[i-1];
            matsiz = iwork[i] - iwork[i-1];
        }
        lapackf77_dsteqr( "I", &matsiz, &d[submat], &e[submat],
                          Q(submat, submat), &ldq, work, info );  // change to edc?
        if (*info != 0) {
            printf( "info: %lld\n, submat: %lld\n", (long long) *info, (long long) submat );
            *info = (submat+1)*(n+1) + submat + matsiz;
            printf( "info: %lld\n", (long long) *info );
            return *info;
        }
        k = 1;
        for (j = submat; j < iwork[i]; ++j) {
            iwork[indxq+j] = k;
            ++k;
        }
    }

    timer_stop( time );
    timer_printf( "  for: dsteqr = %6.2f\n", time );
    
    // Successively merge eigensystems of adjacent submatrices
    // into eigensystem for the corresponding larger matrix.

    curlvl = 1;
    while (subpbs > 1) {
        timer_start( time );
        
        for (i=0; i < subpbs-1; i += 2) {
            if (i == 0) {
                submat = 0;
                matsiz = iwork[1];
                msd2 = iwork[0];
            } else {
                submat = iwork[i-1];
                matsiz = iwork[i+1] - iwork[i-1];
                msd2 = matsiz / 2;
            }

            // Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
            // into an eigensystem of size MATSIZ.
            // DLAEX1 is used only for the full eigensystem of a tridiagonal
            // matrix.

            if (matsiz == n)
                range2 = range;
            else
                // We need all the eigenvectors if it is not last step
                range2 = MagmaRangeAll;

            magma_dlaex1_m( ngpu, matsiz, &d[submat], Q(submat, submat), ldq,
                            &iwork[indxq+submat], e[submat+msd2-1], msd2,
                            work, &iwork[subpbs], dw, queues,
                            range2, vl, vu, il, iu, info );

            if (*info != 0) {
                *info = (submat+1)*(n+1) + submat + matsiz;
                return *info;
            }
            iwork[i/2]= iwork[i+1];
        }
        subpbs /= 2;
        ++curlvl;
        
        timer_stop( time );
        //timer_printf( "%lld: time: %6.2f\n", (long long) curlvl, time );
    }

    // Re-merge the eigenvalues/vectors which were deflated at the final
    // merge step.

    for (i = 0; i < n; ++i) {
        j = iwork[indxq+i] - 1;
        work[i] = d[j];
        blasf77_dcopy( &n, Q(0, j), &ione, &work[ n*(i+1) ], &ione );
    }
    blasf77_dcopy( &n, work, &ione, d, &ione );
    lapackf77_dlacpy( "A", &n, &n, &work[n], &n, Q, &ldq );

cleanup:
    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        magma_queue_destroy( queues[dev][0] );
        magma_queue_destroy( queues[dev][1] );
        magma_free( dw[dev] );
    }

    magma_setdevice( orig_dev );

    return *info;
} /* magma_dlaex0 */
