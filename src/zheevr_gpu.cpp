/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Raffaele Solca
       @author Azzam Haidar

       @precisions normal z -> c
 
*/
#include "magma_internal.h"

//#define FAST_HEMV

/***************************************************************************//**
    Purpose
    -------
    ZHEEVR computes selected eigenvalues and, optionally, eigenvectors
    of a complex Hermitian matrix T.  Eigenvalues and eigenvectors can
    be selected by specifying either a range of values or a range of
    indices for the desired eigenvalues.

    Whenever possible, ZHEEVR calls ZSTEGR to compute the
    eigenspectrum using Relatively Robust Representations.  ZSTEGR
    computes eigenvalues by the dqds algorithm, while orthogonal
    eigenvectors are computed from various "good" L D L^T representations
    (also known as Relatively Robust Representations). Gram-Schmidt
    orthogonalization is avoided as far as possible. More specifically,
    the various steps of the algorithm are as follows. For the i-th
    unreduced block of T,
       1.  Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
            is a relatively robust representation,
       2.  Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
           relative accuracy by the dqds algorithm,
       3.  If there is a cluster of close eigenvalues, "choose" sigma_i
           close to the cluster, and go to step (a),
       4.  Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
           compute the corresponding eigenvector by forming a
           rank-revealing twisted factorization.
    The desired accuracy of the output can be specified by the input
    parameter ABSTOL.

    For more details, see "A new O(n^2) algorithm for the symmetric
    tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
    Computer Science Division Technical Report No. UCB//CSD-97-971,
    UC Berkeley, May 1997.


    Note 1 : ZHEEVR calls ZSTEGR when the full spectrum is requested
    on machines which conform to the ieee-754 floating point standard.
    ZHEEVR calls DSTEBZ and ZSTEIN on non-ieee machines and
    when partial spectrum requests are made.

    Normal execution of ZSTEGR may create NaNs and infinities and
    hence may abort due to a floating point exception in environments
    which do not handle NaNs and infinities in the ieee standard default
    manner.

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
    dA      COMPLEX_16 array, dimension (LDDA, N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, DA is destroyed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

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

    @param[in]
    abstol  DOUBLE PRECISION
            The absolute error tolerance for the eigenvalues.
            An approximate eigenvalue is accepted as converged
            when it is determined to lie in an interval [a,b]
            of width less than or equal to

                    ABSTOL + EPS * max( |a|,|b| ),
    \n
            where EPS is the machine precision.  If ABSTOL is less than
            or equal to zero, then  EPS*|T|  will be used in its place,
            where |T| is the 1-norm of the tridiagonal matrix obtained
            by reducing A to tridiagonal form.
    \n
            See "Computing Small Singular Values of Bidiagonal Matrices
            with Guaranteed High Relative Accuracy," by Demmel and
            Kahan, LAPACK Working Note #3.
    \n
            If high relative accuracy is important, set ABSTOL to
            DLAMCH( 'Safe minimum' ).  Doing so will guarantee that
            eigenvalues are computed to high relative accuracy when
            possible in future releases.  The current code does not
            make any guarantees about high relative accuracy, but
            furutre releases will. See J. Barlow and J. Demmel,
            "Computing Accurate Eigensystems of Scaled Diagonally
            Dominant Matrices", LAPACK Working Note #7, for a discussion
            of which matrices define their eigenvalues to high relative
            accuracy.

    @param[out]
    m       INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = MagmaRangeAll, M = N, and if RANGE = MagmaRangeI, M = IU-IL+1.

    @param[out]
    w       DOUBLE PRECISION array, dimension (N)
            The first M elements contain the selected eigenvalues in
            ascending order.

    @param[out]
    dZ      COMPLEX_16 array, dimension (LDDZ, max(1,M))
            If JOBZ = MagmaVec, then if INFO = 0, the first M columns of Z
            contain the orthonormal eigenvectors of the matrix A
            corresponding to the selected eigenvalues, with the i-th
            column of Z holding the eigenvector associated with W(i).
            If JOBZ = MagmaNoVec, then Z is not referenced.
            Note: the user must ensure that at least max(1,M) columns are
            supplied in the array Z; if RANGE = MagmaRangeV, the exact value of M
            is not known in advance and an upper bound must be used.
*******     (workspace) If FAST_HEMV is defined DZ should be (LDDZ, max(1,N)) in both cases.

    @param[in]
    lddz    INTEGER
            The leading dimension of the array Z.  LDDZ >= 1, and if
            JOBZ = MagmaVec, LDDZ >= max(1,N).

    @param[out]
    isuppz  INTEGER ARRAY, dimension ( 2*max(1,M) )
            The support of the eigenvectors in Z, i.e., the indices
            indicating the nonzero elements in Z. The i-th eigenvector
            is nonzero only in elements ISUPPZ( 2*i-1 ) through
            ISUPPZ( 2*i ).
            __Implemented only for__ RANGE = MagmaRangeAll or MagmaRangeI and IU - IL = N - 1

    @param
    wA      (workspace) COMPLEX_16 array, dimension (LDWA, N)

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.  LDWA >= max(1,N).

    @param
    wZ      (workspace) COMPLEX_16 array, dimension (LDWZ, max(1,M))

    @param[in]
    ldwz    INTEGER
            The leading dimension of the array wZ.  LDWZ >= 1, and if
            JOBZ = MagmaVec, LDWZ >= max(1,N).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK. LWORK >= (NB+1)*N,
            where NB is the max of the blocksize for ZHETRD
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    rwork   (workspace) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal
            (and minimal) LRWORK.

    @param[in]
    lrwork  INTEGER
            The length of the array RWORK.  LRWORK >= max(1,24*N).
    \n
            If LRWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the RWORK array, returns
            this value as the first entry of the RWORK array, and no error
            message related to LRWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK[0] returns the optimal
            (and minimal) LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.  LIWORK >= max(1,10*N).
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  Internal error

    Further Details
    ---------------
    Based on contributions by
       Inderjit Dhillon, IBM Almaden, USA
       Osni Marques, LBNL/NERSC, USA
       Ken Stanley, Computer Science Division, University of
         California at Berkeley, USA

    @ingroup magma_heevr
*******************************************************************************/
extern "C" magma_int_t
magma_zheevr_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
    double *w,
    magmaDoubleComplex_ptr dZ, magma_int_t lddz, magma_int_t *isuppz,
    magmaDoubleComplex *wA,    magma_int_t ldwa,
    magmaDoubleComplex *wZ,    magma_int_t ldwz,
    magmaDoubleComplex *work,  magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    /* Constants */
    magma_int_t ione = 1;
    float szero = 0.;
    float sone  = 1.;
    
    /* Local variables */
    const char* uplo_  = lapack_uplo_const( uplo  );
    const char* jobz_  = lapack_vec_const( jobz  );
    const char* range_ = lapack_range_const( range );
    
    magma_int_t indrd, indre;
    magma_int_t imax;
    magma_int_t lopt, itmp1, indree, indrdd;
    magma_int_t tryrac;
    magma_int_t i, j, jj, i__1;
    magma_int_t iscale, indibl, indifl;
    magma_int_t indiwo, indisp, indtau;
    magma_int_t indrwk, indwk;
    magma_int_t llwork, llrwork, nsplit;
    magma_int_t ieeeok;
    magma_int_t iinfo;
    magma_int_t lwmin, lrwmin, liwmin;
    double safmin;
    double bignum;
    double smlnum;
    double eps, tmp1;
    double anrm;
    double sigma, d__1;
    double rmin, rmax;
    magmaDouble_ptr dwork;
    
    bool lower  = (uplo == MagmaLower);
    bool wantz  = (jobz == MagmaVec);
    bool alleig = (range == MagmaRangeAll);
    bool valeig = (range == MagmaRangeV);
    bool indeig = (range == MagmaRangeI);
    bool lquery = (lwork == -1 || lrwork == -1 || liwork == -1);
    
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
    } else if (lddz < 1 || (wantz && lddz < n)) {
        *info = -15;
    } else if (ldwa < max(1,n)) {
        *info = -18;
    } else if (ldwz < 1 || (wantz && ldwz < n)) {
        *info = -20;
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
    
    magma_int_t nb = magma_get_zhetrd_nb(n);
    
    lwmin =  n * (nb + 1);
    lrwmin = 24 * n;
    liwmin = 10 * n;
    
    work[0] = magma_zmake_lwork( lwmin );
    rwork[0] = magma_dmake_lwork( lrwmin );
    iwork[0] = liwmin;
    
    if (lwork < lwmin && ! lquery) {
        *info = -22;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -24;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -26;
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    } else if (lquery) {
        return *info;
    }
    
    *m = 0;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /* Check if matrix is very small then just call LAPACK on CPU, no need for GPU */
    if (n <= 128) {
        #ifdef ENABLE_DEBUG
        printf("--------------------------------------------------------------\n");
        printf("  warning matrix too small N=%lld NB=%lld, calling lapack on CPU\n", (long long) n, (long long) nb );
        printf("--------------------------------------------------------------\n");
        #endif
        magmaDoubleComplex *A;
        magma_zmalloc_cpu( &A, n*n );
        magma_zgetmatrix( n, n, dA, ldda, A, n, queue );
        lapackf77_zheevr(jobz_, range_, uplo_,
                         &n, A, &n, &vl, &vu, &il, &iu, &abstol, m,
                         w, wZ, &ldwz, isuppz, work, &lwork,
                         rwork, &lrwork, iwork, &liwork, info);
        magma_zsetmatrix( n,  n,  A,    n, dA, ldda, queue );
        magma_zsetmatrix( n, *m, wZ, ldwz, dZ, lddz, queue );
        magma_free_cpu( A );
        magma_queue_destroy( queue );
        return *info;
    }

    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, n )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    --w;
    --work;
    --rwork;
    --iwork;
    --isuppz;
    
    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt(smlnum);
    rmax = magma_dsqrt(bignum);
    
    /* Scale matrix to allowable range, if necessary. */
    anrm = magmablas_zlanhe( MagmaMaxNorm, uplo, n, dA, ldda, dwork, n, queue );
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
        d__1 = 1.;
        magmablas_zlascl( uplo, 0, 0, 1., sigma, n, n, dA, ldda, queue, info );
        
        if (abstol > 0.) {
            abstol *= sigma;
        }
        if (valeig) {
            vl *= sigma;
            vu *= sigma;
        }
    }
    
    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */
    indtau = 1;
    indwk = indtau + n;
    
    indre = 1;
    indrd = indre + n;
    indree = indrd + n;
    indrdd = indree + n;
    indrwk = indrdd + n;
    llwork = lwork - indwk + 1;
    llrwork = lrwork - indrwk + 1;
    
    indifl = 1;
    indibl = indifl + n;
    indisp = indibl + n;
    indiwo = indisp + n;

#ifdef FAST_HEMV
    magma_zhetrd2_gpu(uplo, n, dA, ldda, &rwork[indrd], &rwork[indre],
                      &work[indtau], wA, ldwa, &work[indwk], llwork, dZ, lddz*n, &iinfo);
#else
    magma_zhetrd_gpu (uplo, n, dA, ldda, &rwork[indrd], &rwork[indre],
                      &work[indtau], wA, ldwa, &work[indwk], llwork, &iinfo);
#endif

    lopt = n + (magma_int_t)MAGMA_Z_REAL(work[indwk]);
    
    /* If all eigenvalues are desired and ABSTOL is less than or equal to
       zero, then call DSTERF
       or ZUNGTR and ZSTEQR.  If this fails for
       some eigenvalue, then try DSTEBZ. */
    ieeeok = lapackf77_ieeeck( &ione, &szero, &sone);
    
    /* If only the eigenvalues are required call DSTERF for all or DSTEBZ for a part */
    if (! wantz) {
        blasf77_dcopy(&n, &rwork[indrd], &ione, &w[1], &ione);
        i__1 = n - 1;
        if (alleig || (indeig && il == 1 && iu == n)) {
            lapackf77_dsterf(&n, &w[1], &rwork[indre], info);
            *m = n;
        } else {
            lapackf77_dstebz(range_, "E", &n, &vl, &vu, &il, &iu, &abstol,
                             &rwork[indrd], &rwork[indre], m,
                             &nsplit, &w[1], &iwork[indibl], &iwork[indisp],
                             &rwork[indrwk], &iwork[indiwo], info);
        }
        
        /* Otherwise call ZSTEMR if infinite and NaN arithmetic is supported */
    }
    else if (ieeeok == 1) {
        //printf("MRRR\n");
        i__1 = n - 1;
        
        blasf77_dcopy(&i__1, &rwork[indre], &ione, &rwork[indree], &ione);
        blasf77_dcopy(&n, &rwork[indrd], &ione, &rwork[indrdd], &ione);
        
        if (abstol < 2*n*eps)
            tryrac=1;
        else
            tryrac=0;
        
        lapackf77_zstemr(jobz_, range_, &n, &rwork[indrdd], &rwork[indree], &vl, &vu, &il,
                         &iu, m, &w[1], wZ, &ldwz, &n, &isuppz[1], &tryrac, &rwork[indrwk],
                         &llrwork, &iwork[1], &liwork, info);
        
        if (*info == 0 && wantz) {
            magma_zsetmatrix( n, *m, wZ, ldwz, dZ, lddz, queue );
            magma_zunmtr_gpu(MagmaLeft, uplo, MagmaNoTrans, n, *m, dA, ldda, &work[indtau],
                             dZ, lddz, wA, ldwa, &iinfo);
        }
    }
    
    
    /* Call DSTEBZ and ZSTEIN if infinite and NaN arithmetic is not supported or ZSTEMR didn't converge. */
    if (wantz && (ieeeok == 0 || *info != 0)) {
        //printf("B/I\n");
        *info = 0;
        
        lapackf77_dstebz(range_, "B", &n, &vl, &vu, &il, &iu, &abstol, &rwork[indrd], &rwork[indre], m,
                         &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &rwork[indrwk], &iwork[indiwo], info);
        
        lapackf77_zstein(&n, &rwork[indrd], &rwork[indre], m, &w[1], &iwork[indibl], &iwork[indisp],
                           wZ, &ldwz, &rwork[indrwk], &iwork[indiwo], &iwork[indifl], info);
        
        /* Apply unitary matrix used in reduction to tridiagonal
           form to eigenvectors returned by ZSTEIN. */
        magma_zsetmatrix( n, *m, wZ, ldwz, dZ, lddz, queue );
        magma_zunmtr_gpu(MagmaLeft, uplo, MagmaNoTrans, n, *m, dA, ldda, &work[indtau],
                         dZ, lddz, wA, ldwa, &iinfo);
    }
    
    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_dscal(&imax, &d__1, &w[1], &ione);
    }
    
    /* If eigenvalues are not in order, then sort them, along with
       eigenvectors. */
    if (wantz) {
        for (j = 1; j <= *m-1; ++j) {
            i = 0;
            tmp1 = w[j];
            for (jj = j + 1; jj <= *m; ++jj) {
                if (w[jj] < tmp1) {
                    i = jj;
                    tmp1 = w[jj];
                }
            }
            
            if (i != 0) {
                itmp1 = iwork[indibl + i - 1];
                w[i] = w[j];
                iwork[indibl + i - 1] = iwork[indibl + j - 1];
                w[j] = tmp1;
                iwork[indibl + j - 1] = itmp1;
                magma_zswap( n, dZ + (i-1)*lddz, ione, dZ + (j-1)*lddz, ione, queue );
            }
        }
    }
    
    /* Set WORK[0] to optimal complex workspace size. */
    work[1] = magma_zmake_lwork( lopt );
    rwork[1] = magma_dmake_lwork( lrwmin );
    iwork[1] = liwmin;
    
    magma_queue_destroy( queue );
    magma_free( dwork );

    return *info;
} /* magma_zheevr_gpu */
