/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

*/
#include "magma_internal.h"
#define ENABLE_TIMER
#include "magma_timer.h"

#define REAL

/***************************************************************************//**
    Purpose
    -------
    DSYEVD_2STAGE computes all eigenvalues and, optionally, eigenvectors of a
    real symmetric matrix A. It uses a two-stage algorithm for the tridiagonalization.
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
    A       DOUBLE PRECISION array, dimension (LDA, N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the
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
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
     -      If N <= 1,                      LWORK >= 1.
     -      If JOBZ = MagmaNoVec and N > 1, LWORK >= LWSTG2 + N + N*NB.
     -      If JOBZ = MagmaVec   and N > 1, LWORK >= LWSTG2 + 2*N + N**2.
            where LWSTG2 is the size needed to store the matrices of stage 2
            and is returned by magma_dbulge_getlwstg2.
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

struct sformQ_thread_data {
    magma_int_t n;
    magma_int_t ldda;
    magma_int_t nb;
    magma_int_t Vblksiz;
    magma_queue_t queue;
    float* floatA;
    float* dfloatA;
    float* dfloatQ;
    float* V2;
    magma_int_t ldv;
    float* T2;
    magma_int_t ldt;
    magma_int_t* info;
    float* dT1;

    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    bool* ptr_is_stage2_finished;
};

void *sformQ_GPU(void* threadarg) {
    struct sformQ_thread_data* data = (struct sformQ_thread_data*)threadarg;
    magma_int_t n=data->n;
    magma_int_t ldda=data->ldda;
    magma_int_t nb=data->nb;
    magma_int_t Vblksiz=data->Vblksiz;
    magma_queue_t queue=data->queue;
    float* floatA=data->floatA;
    float* dfloatA=data->dfloatA;
    float* dfloatQ=data->dfloatQ;
    float* V2=data->V2;
    magma_int_t ldv=data->ldv;
    float* T2=data->T2;
    magma_int_t ldt=data->ldt;
    magma_int_t* info=data->info;
    float* dT1=data->dT1;

    pthread_mutex_t* mutex=data->mutex;
    pthread_cond_t* cond=data->cond;
    bool* ptr_is_stage2_finished=data->ptr_is_stage2_finished;

    magma_timer_t time=0;
    timer_start( time );
    magmablas_slaset( MagmaFull, n, n, MAGMA_S_ZERO, MAGMA_S_ONE, dfloatQ, ldda, queue );
    magma_ssetmatrix( n, n, floatA, n, dfloatA, ldda, queue );
    magma_sormqr_2stage_gpu( MagmaLeft, MagmaNoTrans, n-nb, n, n-nb, dfloatA+nb, ldda,
                             dfloatQ+nb, ldda, dT1, nb, info );
    magma_queue_sync( queue );
    magma_free( dfloatA );
    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time 1st back = %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );
    pthread_mutex_lock(mutex);
    while( *ptr_is_stage2_finished==0 ) {
        pthread_cond_wait(cond, mutex);
    }
    pthread_mutex_unlock(mutex);
    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time wait for second stage to complete = %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    magma_sbulge_applyQ_v2(MagmaRight, n, n, nb, Vblksiz, dfloatQ, ldda, V2, ldv, T2, ldt, info);

    magma_queue_sync( queue );
    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time 2nd back = %6.2f\n", (long long) n, (long long) nb, time );
    pthread_exit(NULL);
}



extern "C" magma_int_t
magma_dssyevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *W,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    #define A( i_,j_) (A  + (i_) + (j_)*lda)
    #define floatA( i_,j_) (floatA  + (i_) + (j_)*n)
    #define A2(i_,j_) (A2 + (i_) + (j_)*lda2)
    
    const char* uplo_  = lapack_uplo_const( uplo  );
    const char* jobz_  = lapack_vec_const( jobz  );
    const char* range_ = lapack_range_const( range );
    double c_one  = MAGMA_D_ONE;
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
    magma_int_t nb               = magma_get_dbulge_nb(n, parallel_threads);
    magma_int_t lwstg2           = magma_dbulge_getlwstg2( n, parallel_threads, wantz, 
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

    work[0]  = magma_dmake_lwork( lwmin );
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
        lwmin  = lwstg2 + 1 + 6*n + max(lwstg1, 2*n*n) + n*n + 2*n;
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
        W[0] = MAGMA_D_REAL(A[0]);
        if (wantz) {
            A[0] = MAGMA_D_ONE;
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
        double* Y;
        magma_dmalloc_cpu(&lapack_rwork, 7*n);
        magma_imalloc_cpu(&lapack_iwork, 5*n);
        magma_imalloc_cpu(&ifail, n);
        magma_dmalloc_cpu(&Y, n*ldy);
        lapackf77_dsyevx(jobz_, range_, uplo_,
                         &n, A, &lda, &vl, &vu, &il, &iu, &abstol, m,
                         W, Y, &ldy, work, &lwork,
                         #ifdef COMPLEX
                         lapack_rwork,
                         #endif
                         lapack_iwork, ifail, info);
        if( wantz ) {
            lapackf77_dlacpy(MagmaFullStr, &n, m, Y, &ldy, A, &lda);
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
    anrm = lapackf77_dlansy("M", uplo_, &n, A, &lda, rwork);
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
        lapackf77_dlascl(uplo_, &izero, &izero, &d_one, &sigma, &n, &n, A,
                         &lda, info);
    }

    double *e     = work;
    double *d     = e+n;
    float *floatd = (float*)d; // Same space as d
    float *floate = (float*)(d+n);
    float *floatA = floate+n;
    float *TAU1  = floatA+n*n;
    float *TAU2  = TAU1 + sizTAU1;
    float *V2    = TAU2 + sizTAU2;
    float *T2    = V2   + sizV2;
    float *Wstg1 = T2   + sizT2;
    // PAY ATTENTION THAT work[indA2] should be able to be of size lda2*n
    // which it should be checked in any future modification of lwork.*/
    float *A2    = Wstg1;
    double *Z     = (double*)Wstg1;
    #ifdef COMPLEX
    double *Wedc              = E + n; 
    magma_int_t lwedc         = 1 + 4*n + 2*n*n; // lrwork - n; //used only for wantz>0
    #else
    double *Wedc              = Z + n*n;
    magma_int_t lwedc         = 1 + 4*n + n*n; // lwork - indWEDC; //used only for wantz>0
    #endif

    double *dA, *dX, *dzwork;
    float *dfloatA, *dfloatQ;
    magma_int_t ldda = magma_roundup( n, 32 );

    bool is_stage2_finished=0;
    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER ;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    struct sformQ_thread_data argdata;

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    magma_timer_t time=0, time_total=0;
    timer_start( time_total );
    timer_start( time );

    if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    if (MAGMA_SUCCESS != magma_smalloc( &dfloatA, n*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    if (MAGMA_SUCCESS != magma_smalloc( &dfloatQ, n*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_dsetmatrix( n, n, A, lda, dA, ldda, queue );
    magmablas_dlag2s( n, n, dA, ldda, dfloatA, ldda, queue, info );
    magma_sgetmatrix( n, n, dfloatA, ldda, floatA, n, queue);

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time cast to float= %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    float *dT1;
    if (MAGMA_SUCCESS != magma_smalloc( &dT1, n*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    magma_ssytrd_sy2sb(uplo, n, nb, floatA, lda, TAU1, Wstg1, lwstg1, dT1, info);

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time dsytrd_sy2sb= %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    /* copy the input matrix into WORK(INDWRK) with band storage */
    memset(A2, 0, n*lda2*sizeof(float));

    for (magma_int_t j = 0; j < n-nb; j++) {
        len = nb+1;
        blasf77_scopy( &len, floatA(j,j), &ione, A2(0,j), &ione );
        memset(floatA(j,j), 0, (nb+1)*sizeof(float));
        *floatA(nb+j,j) = 1.0;
    }
    for (magma_int_t j = 0; j < nb; j++) {
        len = nb-j;
        blasf77_scopy( &len, floatA(j+n-nb,j+n-nb), &ione, A2(0,j+n-nb), &ione );
        memset(floatA(j+n-nb,j+n-nb), 0, (nb-j)*sizeof(float));
    }

    // Setup pthread to start early back transforms.
    argdata.n = n;
    argdata.ldda = ldda;
    argdata.nb = nb;
    argdata.Vblksiz = Vblksiz;
    argdata.queue = queue;
    argdata.floatA = floatA;
    argdata.dfloatA = dfloatA;
    argdata.dfloatQ = dfloatQ;
    argdata.V2 = V2;
    argdata.ldv = ldv;
    argdata.T2 = T2;
    argdata.ldt = ldt;
    argdata.info = info;
    argdata.dT1 = dT1;
    argdata.dfloatQ = dfloatQ;
    argdata.mutex = &mutex;
    argdata.cond = &cond;
    argdata.ptr_is_stage2_finished = &is_stage2_finished;
    pthread_create(&thread, &attr, sformQ_GPU, (void *)&argdata);

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time dsytrd_convert = %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );

    magma_ssytrd_sb2st(uplo, n, nb, Vblksiz, A2, lda2, floatd, floate, V2, ldv, TAU2, wantz, T2, ldt);

    // Send pthread signal
    pthread_mutex_lock(&mutex);
    is_stage2_finished = 1;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);

    lapackf77_slag2d(&n, &ione, floatd, &lda, W, &lda, info);
    lapackf77_dlacpy("A", &n, &ione, W, &lda, d, &lda);
    lapackf77_slag2d(&n, &ione, floate, &lda, e, &lda, info);

    timer_stop( time );
    timer_stop( time_total );
    timer_printf( "  N= %10lld  nb= %5lld time dsytrd_sb2st= %6.2f\n", (long long) n, (long long) nb, time );
    timer_printf( "  N= %10lld  nb= %5lld time dsytrd= %6.2f\n", (long long) n, (long long) nb, time_total );

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call DORMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */
    timer_start( time_total );
        
    double* dwedc;
    if (MAGMA_SUCCESS != magma_dmalloc( &dwedc, 3*n*(n/2 + 1) )) {
        // TODO free dT1, etc. --- see goto cleanup in dlaex0_m.cpp, etc.
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    timer_start( time );

    magma_dstedx(range, n, vl, vu, il, iu, W, e,
                 Z, ldz, Wedc, lwedc,
                 iwork, liwork, dwedc, info);


    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time dstedx = %6.2f\n", (long long) n, (long long) nb, time );
    timer_start( time );
    magma_free( dwedc );
    magma_dmove_eig(range, n, W, &il, &iu, vl, vu, m);

    magma_int_t ldr = ldda;
    if (MAGMA_SUCCESS != magma_dmalloc( &dX, *m * ldr )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    magma_int_t ldzwork = 11 * (*m) * ldr + 4 * ldr;
    if (MAGMA_SUCCESS != magma_dmalloc( &dzwork, ldzwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    lapackf77_slag2d(&n, &ione, floate, &lda, e, &lda, info);
    magma_dsetmatrix( n, *m, Z +ldz * (il-1), n, dX, ldr, queue);
    magmablas_dsymmetrize( uplo, n, dA, ldda, queue );
    pthread_join(thread, NULL);
    magma_dssysicesm( n, *m, dA, ldda, dfloatQ, ldr, dX, ldr, W, d, e,
                      dzwork, ldzwork, queue);
    magma_dgetmatrix( n, *m, dX, ldr, A, lda, queue);

    timer_stop( time );
    timer_printf( "  N= %10lld  nb= %5lld time iterative refinement = %6.2f\n",
                  (long long) n, (long long) nb, time );
    magma_free(dA);
    magma_free(dfloatQ);
    magma_free(dX);
    magma_free(dzwork);
    magma_free(dT1);
    magma_queue_destroy( queue );
    timer_stop( time_total );
    timer_printf( "  N= %10lld  nb= %5lld time eigenvectors backtransf. = %6.2f\n", 
                  (long long) n, (long long) nb, time_total );
    
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

    work[0]  = magma_dmake_lwork( lwmin );
    #ifdef COMPLEX
    rwork[0] = magma_dmake_lwork( lrwmin );
    #endif
    iwork[0] = liwmin;

    return *info;
} /* magma_dsyevdx_2stage */
