/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

       @precisions normal z -> c d s

*/
#include "magma_internal.h"
#include "magma_timer.h"

#define COMPLEX

/***************************************************************************//**
    Purpose
    -------
    ZHEEVD_2STAGE computes all eigenvalues and, optionally, eigenvectors of a
    complex Hermitian matrix A. It uses a two-stage algorithm for the tridiagonalization.
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
    A       COMPLEX_16 array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = MagmaVec, then if INFO = 0, the first m columns
            of A contains the required
            orthonormal eigenvectors of the matrix A.
            If JOBZ = MagmaNoVec, then on exit the lower triangle (if UPLO=MagmaLower)
            or the upper triangle (if UPLO=MagmaUpper) of A, including the
            diagonal, is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

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
    m       INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = MagmaRangeAll, M = N, and if RANGE = MagmaRangeI, M = IU-IL+1.

    @param[out]
    W       DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the required m eigenvalues in ascending order.

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
     -      If N <= 1,                      LWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LWORK >= LWSTG2 + N + N*NB.
     -      If JOBZ = MagmaVec   and N > 1, LWORK >= LWSTG2 + 2*N + N**2.
            where LWSTG2 is the size needed to store the matrices of stage 2
            and is returned by magma_zbulge_getlwstg2.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

*/
#ifdef COMPLEX
/**

    @param[out]
    rwork   (workspace) DOUBLE PRECISION array,
                                           dimension (LRWORK)
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

*/
#endif
/**

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

    @ingroup magma_heevdx
*******************************************************************************/
extern "C" magma_int_t
magma_zheevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *W,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    #define A( i_,j_) (A  + (i_) + (j_)*lda)
    #define A2(i_,j_) (A2 + (i_) + (j_)*lda2)
    
    const char* uplo_  = lapack_uplo_const( uplo  );
    const char* jobz_  = lapack_vec_const( jobz  );
    const char* range_ = lapack_range_const( range );
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magma_int_t ione = 1;
    magma_int_t izero = 0;
    double d_one = 1.;

    double d__1;

    double eps;
    double anrm;
    magma_int_t imax;
    double rmin, rmax;
    double sigma;
    #ifdef COMPLEX
    magma_int_t lrwmin;
    #endif
    magma_int_t lwmin, liwmin;
    magma_int_t lower;
    magma_int_t wantz;
    magma_int_t iscale;
    double safmin;
    double bignum;
    double smlnum;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;
    magma_int_t len;

    wantz  = (jobz == MagmaVec);
    lower  = (uplo == MagmaLower);
    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);

    /* determine the number of threads and other parameter */
    magma_int_t Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2, sizTAU1, ldz, lwstg1, lda2;
    magma_int_t parallel_threads = magma_get_parallel_numthreads();
    magma_int_t nb               = magma_get_zbulge_nb(n, parallel_threads);
    magma_int_t lwstg2           = magma_zbulge_getlwstg2( n, parallel_threads, wantz, 
                                                           &Vblksiz, &ldv, &ldt, &blkcnt, 
                                                           &sizTAU2, &sizT2, &sizV2);
    // lwstg1=nb*n but since used also to store the band A2 so it is 2nb*n;
    lwstg1                       = magma_bulge_getlwstg1( n, nb, &lda2 );

    sizTAU1                      = n;
    ldz                          = n;

    #ifdef COMPLEX
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);
    #else
    lquery = (lwork == -1 || liwork == -1);
    #endif

    *info = 0;
    if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,n)) {
        *info = -6;
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


    #ifdef COMPLEX
    if (wantz) {
        lwmin  = lwstg2 + 2*n + max(lwstg1, n*n);
        lrwmin = 1 + 5*n + 2*n*n;
        liwmin = 5*n + 3;
    } else {
        lwmin  = lwstg2 + n + lwstg1;
        lrwmin = n;
        liwmin = 1;
    }

    work[0]  = magma_zmake_lwork( lwmin );
    rwork[0] = magma_dmake_lwork( lrwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -14;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -16;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -18;
    }
    #else
    if (wantz) {
        lwmin  = lwstg2 + 1 + 6*n + max(lwstg1, 2*n*n);
        liwmin = 5*n + 3;
    } else {
        lwmin  = lwstg2 + 2*n + lwstg1;
        liwmin = 1;
    }

    work[0]  = magma_dmake_lwork( lwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -14;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -16;
    }
    #endif

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (n == 0) {
        return *info;
    }

    if (n == 1) {
        W[0] = MAGMA_Z_REAL(A[0]);
        if (wantz) {
            A[0] = MAGMA_Z_ONE;
        }
        return *info;
    }


    timer_printf("using %lld parallel_threads\n", (long long) parallel_threads );

    /* Check if matrix is very small then just call LAPACK on CPU, no need for GPU */
    magma_int_t ntiles = n/nb;
    if ( ( ntiles < 2 ) || ( n <= 128 ) ) {
        #ifdef ENABLE_DEBUG
        printf("--------------------------------------------------------------\n");
        printf("  warning matrix too small N=%lld NB=%lld, calling lapack on CPU\n", 
               (long long) n, (long long) nb );
        printf("--------------------------------------------------------------\n");
        #endif
        double abstol = 2 * lapackf77_dlamch("Safe minimum");
        magma_int_t ldy = lda;
        double* lapack_rwork;
        magma_int_t* lapack_iwork;
        magma_int_t* ifail;
        magmaDoubleComplex* Y;
        magma_dmalloc_cpu(&lapack_rwork, 7*n);
        magma_imalloc_cpu(&lapack_iwork, 5*n);
        magma_imalloc_cpu(&ifail, n);
        magma_zmalloc_cpu(&Y, n*ldy);
        lapackf77_zheevx(jobz_, range_, uplo_,
                         &n, A, &lda, &vl, &vu, &il, &iu, &abstol, m,
                         W, Y, &ldy, work, &lwork,
                         #ifdef COMPLEX
                         lapack_rwork,
                         #endif
                         lapack_iwork, ifail, info);
        if( wantz ) {
            lapackf77_zlacpy(MagmaFullStr, &n, m, Y, &ldy, A, &lda);
        }
        magma_free_cpu(lapack_rwork);
        magma_free_cpu(lapack_iwork);
        magma_free_cpu(ifail);
        magma_free_cpu(Y);
        return *info;
    }

    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt(smlnum);
    rmax = magma_dsqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */
    #ifdef COMPLEX
    anrm = lapackf77_zlanhe("M", uplo_, &n, A, &lda, rwork);
    #else
    anrm = lapackf77_dlansy("M", uplo_, &n, A, &lda, work);
    #endif
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        lapackf77_zlascl(uplo_, &izero, &izero, &d_one, &sigma, &n, &n, A,
                         &lda, info);
    }

    #ifdef COMPLEX
    double *E                 = rwork;
    magma_int_t sizE_onwork   = 0;
    #else
    double *E                 = work;
    magma_int_t sizE_onwork   = n;
    #endif
    
    magmaDoubleComplex *TAU1  = work + sizE_onwork;
    magmaDoubleComplex *TAU2  = TAU1 + sizTAU1;
    magmaDoubleComplex *V2    = TAU2 + sizTAU2;
    magmaDoubleComplex *T2    = V2   + sizV2;
    magmaDoubleComplex *Wstg1 = T2   + sizT2;
    // PAY ATTENTION THAT work[indA2] should be able to be of size lda2*n
    // which it should be checked in any future modification of lwork.*/
    magmaDoubleComplex *A2    = Wstg1;
    magmaDoubleComplex *Z     = Wstg1;
    #ifdef COMPLEX
    double *Wedc              = E + n; 
    magma_int_t lwedc         = 1 + 4*n + 2*n*n; // lrwork - n; //used only for wantz>0
    #else
    double *Wedc              = Wstg1 + n*n;
    magma_int_t lwedc         = 1 + 4*n + n*n; // lwork - indWEDC; //used only for wantz>0
    #endif


    magma_timer_t time=0, time_total=0;
    timer_start( time_total );
    timer_start( time );

    magmaDoubleComplex *dT1;
    if (MAGMA_SUCCESS != magma_zmalloc( &dT1, n*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    magma_zhetrd_he2hb(uplo, n, nb, A, lda, TAU1, Wstg1, lwstg1, dT1, info);

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time zhetrd_he2hb= %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    /* copy the input matrix into WORK(INDWRK) with band storage */
    memset(A2, 0, n*lda2*sizeof(magmaDoubleComplex));

    for (magma_int_t j = 0; j < n-nb; j++) {
        len = nb+1;
        blasf77_zcopy( &len, A(j,j), &ione, A2(0,j), &ione );
        memset(A(j,j), 0, (nb+1)*sizeof(magmaDoubleComplex));
        *A(nb+j,j) = c_one;
    }
    for (magma_int_t j = 0; j < nb; j++) {
        len = nb-j;
        blasf77_zcopy( &len, A(j+n-nb,j+n-nb), &ione, A2(0,j+n-nb), &ione );
        memset(A(j+n-nb,j+n-nb), 0, (nb-j)*sizeof(magmaDoubleComplex));
    }

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time zhetrd_convert = %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    magma_zhetrd_hb2st(uplo, n, nb, Vblksiz, A2, lda2, W, E, V2, ldv, TAU2, wantz, T2, ldt);

    timer_stop( time );
    timer_stop( time_total );
    timer_printf( "  N= %10lld  nb= %5lld time zhetrd_hb2st= %6.2f\n", (long long) n, (long long) nb, time );
    timer_printf( "  N= %10lld  nb= %5lld time zhetrd= %6.2f\n", (long long) n, (long long) nb, time_total );

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call ZUNMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */
    if (! wantz) {
        timer_start( time );

        lapackf77_dsterf(&n, W, E, info);
        magma_dmove_eig(range, n, W, &il, &iu, vl, vu, m);

        timer_stop( time );
        timer_printf( "  N= %10lld  nb= %5lld time dstedc = %6.2f\n", (long long) n, (long long) nb, time );
    }
    else {
        timer_start( time_total );
        
        double* dwedc;
        if (MAGMA_SUCCESS != magma_dmalloc( &dwedc, 3*n*(n/2 + 1) )) {
            // TODO free dT1, etc. --- see goto cleanup in dlaex0_m.cpp, etc.
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        timer_start( time );

        magma_zstedx(range, n, vl, vu, il, iu, W, E,
                     Z, ldz, Wedc, lwedc,
                     iwork, liwork, dwedc, info);


        timer_stop( time );
        timer_printf( "  N= %10lld  nb= %5lld time zstedx = %6.2f\n", (long long) n, (long long) nb, time );
        magma_free( dwedc );
        magma_dmove_eig(range, n, W, &il, &iu, vl, vu, m);

        magmaDoubleComplex *dZ;
        magma_int_t lddz = n;

        if (MAGMA_SUCCESS != magma_zmalloc( &dZ, (*m)*lddz)) {
            // TODO free dT1, etc. --- see goto cleanup in dlaex0_m.cpp, etc.
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        timer_start( time );

        magma_zbulge_back(uplo, n, nb, *m, Vblksiz, Z +ldz*(il-1), ldz, dZ, lddz,
                          V2, ldv, TAU2, T2, ldt, info);

        timer_stop( time );
        timer_printf( "  N= %10lld  nb= %5lld time zbulge_back = %6.2f\n", (long long) n, (long long) nb, time );

        magmaDoubleComplex *dA;
        magma_int_t ldda = n;
        if (MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda )) {
            // TODO free dT1, etc. --- see goto cleanup in dlaex0_m.cpp, etc.
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        timer_start( time );

        magma_queue_t queue;
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue );

        magma_zsetmatrix( n, n, A, lda, dA, ldda, queue );

        magma_zunmqr_2stage_gpu( MagmaLeft, MagmaNoTrans, n-nb, *m, n-nb, dA+nb, ldda,
                                 dZ+nb, n, dT1, nb, info );

        magma_zgetmatrix( n, *m, dZ, lddz, A, lda, queue );

        magma_queue_sync( queue );
        magma_queue_destroy( queue );

        timer_stop( time );
        timer_printf( "  N= %10lld  nb= %5lld time zunmqr + copy = %6.2f\n", 
                      (long long) n, (long long) nb, time );
        magma_free(dZ);
        magma_free(dA);
        timer_stop( time_total );
        timer_printf( "  N= %10lld  nb= %5lld time eigenvectors backtransf. = %6.2f\n", 
                      (long long) n, (long long) nb, time_total );
    }

    magma_free(dT1);
    
    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_dscal(&imax, &d__1, W, &ione);
    }

    work[0]  = magma_zmake_lwork( lwmin );
    #ifdef COMPLEX
    rwork[0] = magma_dmake_lwork( lrwmin );
    #endif
    iwork[0] = liwmin;

    return *info;
} /* magma_zheevdx_2stage */
