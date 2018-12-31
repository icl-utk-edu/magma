/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Raffaele Solca
       @author Mark Gates
       @author Azzam Haidar

       @precisions normal d -> s

*/
#include "magma_internal.h"
#include "magma_timer.h"

#define REAL
#define FAST_SYMV

/***************************************************************************//**
    Purpose
    -------
    DSYEVDX computes selected eigenvalues and, optionally, eigenvectors
    of a real symmetric matrix A. Eigenvalues and eigenvectors can
    be selected by specifying either a range of values or a range of
    indices for the desired eigenvalues.
    If eigenvectors are desired, it uses a divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                   will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU,
            dimension (LDDA, N).
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = MagmaVec, then if INFO = 0, the first mout columns
            of A contains the required
            orthonormal eigenvectors of the matrix A.
            If JOBZ = MagmaNoVec, then on exit the lower triangle (if UPLO=MagmaLower)
            or the upper triangle (if UPLO=MagmaUpper) of A, including the
            diagonal, is destroyed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.  LDDA >= max(1,N).

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
    mout    INTEGER
            The total number of eigenvalues found.  0 <= MOUT <= N.
            If RANGE = MagmaRangeAll, MOUT = N, and if RANGE = MagmaRangeI, MOUT = IU-IL+1.

    @param[out]
    w       DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the required mout eigenvalues in ascending order.

    @param
    wA      (workspace) DOUBLE PRECISION array, dimension (LDWA, N)

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.  LDWA >= max(1,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
     -      If N <= 1,                      LWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LWORK >= 2*N + N*NB.
     -      If JOBZ = MagmaVec   and N > 1, LWORK >= max( 2*N + N*NB, 1 + 6*N + 2*N**2 ).
            NB can be obtained through magma_get_dsytrd_nb(N).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK and IWORK
            arrays, returns these values as the first entries of the WORK
            and IWORK arrays, and no error message related to LWORK or
            LIWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.
     -      If N <= 1,                      LIWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LIWORK >= 1.
     -      If JOBZ = MagmaVec   and N > 1, LIWORK >= 3 + 5*N.
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK and
            IWORK arrays, returns these values as the first entries of
            the WORK and IWORK arrays, and no error message related to
            LWORK or LIWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i and JOBZ = MagmaNoVec, then the algorithm failed
                  to converge; i off-diagonal elements of an intermediate
                  tridiagonal form did not converge to zero;
                  if INFO = i and JOBZ = MagmaVec, then the algorithm failed
                  to compute an eigenvalue while working on the submatrix
                  lying in rows and columns INFO/(N+1) through
                  mod(INFO,N+1).

    Further Details
    ---------------
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    Modified description of INFO. Sven, 16 Feb 05.

    @ingroup magma_heevdx
*******************************************************************************/
extern "C" magma_int_t
magma_dsyevdx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    double *wA,  magma_int_t ldwa,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    magma_int_t ione = 1;

    double d__1;

    double eps;
    magma_int_t inde;
    double anrm;
    double rmin, rmax;
    double sigma;
    magma_int_t iinfo, lwmin;
    magma_int_t lower;
    magma_int_t wantz;
    magma_int_t indwk2, llwrk2;
    magma_int_t iscale;
    double safmin;
    double bignum;
    magma_int_t indtau;
    magma_int_t indwrk, liwmin;
    magma_int_t llwork;
    double smlnum;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;

    magmaDouble_ptr dwork;
    magma_int_t lddc = ldda;

    wantz = (jobz == MagmaVec);
    lower = (uplo == MagmaLower);

    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);

    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldda < max(1,n)) {
        *info = -6;
    } else if (ldwa < max(1,n)) {
        *info = -14;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -9;
            } else if (iu < min(n,il) || iu > n) {
                *info = -10;
            }
        }
    }

    magma_int_t nb = magma_get_dsytrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = max( 2*n + n*nb, 1 + 6*n + 2*n*n );
        liwmin = 3 + 5*n;
    }
    else {
        lwmin  = 2*n + n*nb;
        liwmin = 1;
    }
    
    work[0]  = magma_dmake_lwork( lwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -16;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -18;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /* If matrix is very small, then just call LAPACK on CPU, no need for GPU */
    if (n <= 128) {
        magma_int_t lda = n;
        double *A;
        magma_dmalloc_cpu( &A, lda*n );
        magma_dgetmatrix( n, n, dA, ldda, A, lda, queue );
        // TODO: deal with range and mout. Use lapack dsyevx? (There is no lapack dsyevdx.)
        lapackf77_dsyevd( lapack_vec_const(jobz), lapack_uplo_const(uplo),
                          &n, A, &lda,
                          w, work, &lwork,
                          iwork, &liwork, info );
        *mout = n;
        magma_dsetmatrix( n, n, A, lda, dA, ldda, queue );
        magma_free_cpu( A );
        magma_queue_destroy( queue );
        return *info;
    }

    // dsytrd2_gpu requires ldda*ceildiv(n,64) + 2*ldda*nb
    // dormtr_gpu  requires lddc*n
    // dlansy      requires n
    magma_int_t ldwork = max( ldda*magma_ceildiv(n,64) + 2*ldda*nb, lddc*n );
    ldwork = max( ldwork, n );
    if ( wantz ) {
        // dstedx requires 3n^2/2
        ldwork = max( ldwork, 3*n*(n/2 + 1) );
    }
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, ldwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps    = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt( smlnum );
    rmax = magma_dsqrt( bignum );

    /* Scale matrix to allowable range, if necessary. */
    anrm = magmablas_dlansy( MagmaMaxNorm, uplo, n, dA, ldda, dwork, ldwork, queue );
    iscale = 0;
    sigma  = 1;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        magmablas_dlascl( uplo, 0, 0, 1., sigma, n, n, dA, ldda, queue, info );
    }

    /* Call DSYTRD to reduce symmetric matrix to tridiagonal form. */
    // dsytrd work: e (n) + tau (n) + llwork (n*nb)  ==>  2n + n*nb
    // dstedx work: e (n) + tau (n) + z (n*n) + llwrk2 (1 + 4*n + n^2)  ==>  1 + 6n + 2n^2
    inde   = 0;
    indtau = inde   + n;
    indwrk = indtau + n;
    indwk2 = indwrk + n*n;
    llwork = lwork - indwrk;
    llwrk2 = lwork - indwk2;

    magma_timer_t time=0;
    timer_start( time );

#ifdef FAST_SYMV
    magma_dsytrd2_gpu( uplo, n, dA, ldda, w, &work[inde],
                       &work[indtau], wA, ldwa, &work[indwrk], llwork,
                       dwork, ldwork, &iinfo );
#else
    magma_dsytrd_gpu(  uplo, n, dA, ldda, w, &work[inde],
                       &work[indtau], wA, ldwa, &work[indwrk], llwork,
                       &iinfo );
#endif

    timer_stop( time );
    timer_printf( "time dsytrd = %6.2f\n", time );

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call DORMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */
    if (! wantz) {
        timer_start( time );
        lapackf77_dsterf( &n, w, &work[inde], info );
        magma_dmove_eig( range, n, w, &il, &iu, vl, vu, mout );
        timer_stop( time );
        timer_printf( "time dsterf = %6.2f\n", time );
    }
    else {
        timer_start( time );

        magma_dstedx( range, n, vl, vu, il, iu, w, &work[inde],
                      &work[indwrk], n, &work[indwk2],
                      llwrk2, iwork, liwork, dwork, info );

        timer_stop( time );
        timer_printf( "time dstedx = %6.2f\n", time );
        timer_start( time );

        magma_dmove_eig( range, n, w, &il, &iu, vl, vu, mout );

        magma_dsetmatrix( n, *mout, &work[indwrk + n* (il-1) ], n, dwork, lddc, queue );

        magma_dormtr_gpu( MagmaLeft, uplo, MagmaNoTrans, n, *mout, dA, ldda, &work[indtau],
                          dwork, lddc, wA, ldwa, &iinfo );

        magma_dcopymatrix( n, *mout, dwork, lddc, dA, ldda, queue );

        timer_stop( time );
        timer_printf( "time dormtr + copy = %6.2f\n", time );
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        d__1 = 1. / sigma;
        blasf77_dscal( &n, &d__1, w, &ione );
    }

    work[0]  = magma_dmake_lwork( lwmin );
    iwork[0] = liwmin;

    magma_queue_destroy( queue );
    magma_free( dwork );

    return *info;
} /* magma_dsyevd_gpu */
