/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar

       @precisions normal z -> c

*/
#include "magma_internal.h"
#include "magma_timer.h"

#define COMPLEX
#define FAST_HEMV

/***************************************************************************//**
    Purpose
    -------
    ZHEEVD_GPU computes all eigenvalues and, optionally, eigenvectors of a
    complex Hermitian matrix A.  If eigenvectors are desired, it uses a
    divide and conquer algorithm.

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
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU,
            dimension (LDDA, N).
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
            orthonormal eigenvectors of the matrix A.
            If JOBZ = MagmaNoVec, then on exit the lower triangle (if UPLO=MagmaLower)
            or the upper triangle (if UPLO=MagmaUpper) of A, including the
            diagonal, is destroyed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.  LDDA >= max(1,N).

    @param[out]
    w       DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    @param
    wA      (workspace) COMPLEX_16 array, dimension (LDWA, N)

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.  LDWA >= max(1,N).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
     -      If N <= 1,                      LWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LWORK >= N + N*NB.
     -      If JOBZ = MagmaVec   and N > 1, LWORK >= max( N + N*NB, 2*N + N**2 ).
            NB can be obtained through magma_get_zhetrd_nb(N).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    rwork   (workspace) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal LRWORK.

    @param[in]
    lrwork  INTEGER
            The dimension of the array RWORK.
     -      If N <= 1,                      LRWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LRWORK >= N.
     -      If JOBZ = MagmaVec   and N > 1, LRWORK >= 1 + 5*N + 2*N**2.
    \n
            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

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
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

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

    @ingroup magma_heevd
*******************************************************************************/
extern "C" magma_int_t
magma_zheevd_gpu(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *w,
    magmaDoubleComplex *wA,  magma_int_t ldwa,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    const char* uplo_ = lapack_uplo_const( uplo );
    const char* jobz_ = lapack_vec_const( jobz );
    magma_int_t ione = 1;

    double d__1;

    double eps;
    magma_int_t inde;
    double anrm;
    magma_int_t imax;
    double rmin, rmax;
    double sigma;
    magma_int_t iinfo, lwmin;
    magma_int_t lower;
    magma_int_t llrwk;
    magma_int_t wantz;
    //magma_int_t indwk2;
    magma_int_t iscale;
    double safmin;
    double bignum;
    magma_int_t indtau;
    magma_int_t indrwk, indwrk, liwmin;
    magma_int_t lrwmin, llwork;
    double smlnum;
    magma_int_t lquery;

    magmaDouble_ptr dwork;
    magmaDoubleComplex_ptr dC;
    magma_int_t lddc = ldda;

    wantz = (jobz == MagmaVec);
    lower = (uplo == MagmaLower);
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -1;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (ldwa < max(1,n)) {
        *info = -8;
    }

    magma_int_t nb = magma_get_zhetrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        lrwmin = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = max( n + n*nb, 2*n + n*n );
        lrwmin = 1 + 5*n + 2*n*n;
        liwmin = 3 + 5*n;
    }
    else {
        lwmin  = n + n*nb;
        lrwmin = n;
        liwmin = 1;
    }
    
    work[0]  = magma_zmake_lwork( lwmin );
    rwork[0] = magma_dmake_lwork( lrwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -10;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -12;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -14;
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
        magmaDoubleComplex *A;
        magma_zmalloc_cpu( &A, lda*n );
        magma_zgetmatrix( n, n, dA, ldda, A, lda, queue );
        lapackf77_zheevd( jobz_, uplo_,
                          &n, A, &lda,
                          w, work, &lwork,
                          rwork, &lrwork,
                          iwork, &liwork, info );
        magma_zsetmatrix( n, n, A, lda, dA, ldda, queue );
        magma_free_cpu( A );
        magma_queue_destroy( queue );
        return *info;
    }

    // dC and dwork are never used together, so use one buffer for both;
    // unfortunately they're different types (complex and double).
    // (this is easier in dsyevd_gpu where everything is double.)
    // zhetrd2_gpu requires ldda*ceildiv(n,64) + 2*ldda*nb, in double-complex.
    // zunmtr_gpu  requires lddc*n,                         in double-complex.
    // zlanhe      requires n, in double.
    magma_int_t ldwork = max( ldda*magma_ceildiv(n,64) + 2*ldda*nb, lddc*n );
    magma_int_t ldwork_real = max( ldwork*2, n );
    if ( wantz ) {
        // zstedx requrise 3n^2/2, in double
        ldwork_real = max( ldwork_real, 3*n*(n/2 + 1) );
    }
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, ldwork_real )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dC = (magmaDoubleComplex*) dwork;

    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps    = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt( smlnum );
    rmax = magma_dsqrt( bignum );

    /* Scale matrix to allowable range, if necessary. */
    anrm = magmablas_zlanhe( MagmaMaxNorm, uplo, n, dA, ldda, dwork, ldwork, queue );
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
        magmablas_zlascl( uplo, 0, 0, 1., sigma, n, n, dA, ldda, queue, info );
    }

    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */
    // zhetrd rwork: e (n)
    // zstedx rwork: e (n) + llrwk (1 + 4*N + 2*N**2)  ==>  1 + 5n + 2n^2
    inde   = 0;
    indrwk = inde + n;
    llrwk  = lrwork - indrwk;

    // zhetrd work: tau (n) + llwork (n*nb)  ==>  n + n*nb
    // zstedx work: tau (n) + z (n^2)
    // zunmtr work: tau (n) + z (n^2) + llwrk2 (n or n*nb)  ==>  2n + n^2, or n + n*nb + n^2
    indtau = 0;
    indwrk = indtau + n;
    //indwk2 = indwrk + n*n;
    llwork = lwork - indwrk;
    //llwrk2 = lwork - indwk2;

    magma_timer_t time=0;
    timer_start( time );

#ifdef FAST_HEMV
    magma_zhetrd2_gpu( uplo, n, dA, ldda, w, &rwork[inde],
                       &work[indtau], wA, ldwa, &work[indwrk], llwork,
                       dC, ldwork, &iinfo );
#else
    magma_zhetrd_gpu(  uplo, n, dA, ldda, w, &rwork[inde],
                       &work[indtau], wA, ldwa, &work[indwrk], llwork,
                       &iinfo );
#endif

    timer_stop( time );
    timer_printf( "time zhetrd_gpu = %6.2f\n", time );

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call ZUNMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */
    if (! wantz) {
        timer_start( time );
        lapackf77_dsterf( &n, w, &rwork[inde], info );
        timer_stop( time );
        timer_printf( "time dsterf = %6.2f\n", time );
    }
    else {
        timer_start( time );

        magma_zstedx( MagmaRangeAll, n, 0., 0., 0, 0, w, &rwork[inde],
                      &work[indwrk], n, &rwork[indrwk],
                      llrwk, iwork, liwork, dwork, info );

        timer_stop( time );
        timer_printf( "time zstedx = %6.2f\n", time );
        timer_start( time );

        magma_zsetmatrix( n, n, &work[indwrk], n, dC, lddc, queue );

        magma_zunmtr_gpu( MagmaLeft, uplo, MagmaNoTrans, n, n, dA, ldda, &work[indtau],
                          dC, lddc, wA, ldwa, &iinfo );

        magma_zcopymatrix( n, n, dC, lddc, dA, ldda, queue );

        timer_stop( time );
        timer_printf( "time zunmtr_gpu + copy = %6.2f\n", time );
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_dscal( &imax, &d__1, w, &ione );
    }

    work[0]  = magma_zmake_lwork( lwmin );
    rwork[0] = magma_dmake_lwork( lrwmin );
    iwork[0] = liwmin;

    magma_queue_destroy( queue );
    magma_free( dwork );

    return *info;
} /* magma_zheevd_gpu */
