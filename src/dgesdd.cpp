/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal d -> s

*/
#include "magma_internal.h"

#define REAL

// Version 1 - LAPACK
// Version 2 - MAGMA
#define VERSION 2

const char* dgesdd_path = "none";

/***************************************************************************//**
    Purpose
    -------
    DGESDD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and right singular
    vectors, by using divide-and-conquer method. The SVD is written

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**T, not V.

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
            Specifies options for computing all or part of the matrix U:
      -     = MagmaAllVec:  all M columns of U and all N rows of V**T are
                            returned in the arrays U and VT;
      -     = MagmaSomeVec: the first min(M,N) columns of U and
                            the first min(M,N) rows of V**T are
                            returned in the arrays U and VT;
      -     = MagmaOverwriteVec:
                    If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**T are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**T are overwritten
                    on the array A;
      -     = MagmaNoVec:   no columns of U or rows of V**T are computed.

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
      -     if JOBZ = MagmaOverwriteVec,
                if M >= N, A is overwritten with the first N columns
                of U (the left singular vectors, stored columnwise);
                otherwise, A is overwritten with the first M rows
                of V**T (the right singular vectors, stored rowwise).
      -     if JOBZ != MagmaOverwriteVec, the contents of A are destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i + 1).

    @param[out]
    U       DOUBLE PRECISION array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N;
            UCOL = min(M,N) if JOBZ = MagmaSomeVec.
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N,
            U contains the M-by-M orthogonal matrix U;
      -     if JOBZ = MagmaSomeVec, U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBZ = MagmaOverwriteVec and M >= N, or JOBZ = MagmaNoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = MagmaSomeVec or MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N, LDU >= M.

    @param[out]
    VT      DOUBLE PRECISION array, dimension (LDVT,N)
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N,
            VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBZ = MagmaSomeVec, VT contains the first min(M,N) rows of
            V**T (the right singular vectors, stored rowwise);
      -     if JOBZ = MagmaOverwriteVec and M < N, or JOBZ = MagmaNoVec, VT is not referenced.

    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N, LDVT >= N;
            if JOBZ = MagmaSomeVec, LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,lwork))
            On exit, if INFO = 0, WORK[0] returns the optimal lwork.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= mn*11/6.
            For job: N=None, O=Overwrite, S=Some, A=All.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_dgesvd_nb(M,N).
    \n
            Optimal lwork (required in MAGMA)
            for mx >> mn:
            Path 1:   jobz=N  3*mn + 2*mn*nb
            Path 2:   jobz=O  2*mn*mn       + 3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb )
                          or  mx*mn + mn*mn + 3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb )  [marginally faster?]
            Path 3:   jobz=S  mn*mn +         3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb )
            Path 4:   jobz=A  mn*mn +                max( 3*mn*mn + 7*mn, 3*mn + 2*mn*nb, mn + mx*nb )
            for mx >= mn, but not mx >> mn:
            Path 5:   jobz=N          3*mn + max( 4*mn,           (mx + mn)*nb )
                      jobz=O  mx*mn + 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb )  [faster algorithm]
                          or  mn*mn + 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb )  [slower algorithm]
                      jobz=S          3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb )
                      jobz=A          3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb )
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork
            for mx >> mn:
            Path 1:   jobz=N  8*mn
            Path 2:   jobz=O  2*mn*mn + 3*mn + (3*mn*mn + 4*mn)
            Path 3:   jobz=S    mn*mn + 3*mn + (3*mn*mn + 4*mn)
            Path 4:   jobz=A    mn*mn + 2*mn + mx + (3*mn*mn + 4*mn)    # LAPACK's overestimate
                         or     mn*mn + max( 3*mn*mn + 7*mn, mn + mx )  # correct minimum
            for mx >= mn, but not mx >> mn:
            Path 5:   jobz=N   3*mn + max( 7*mn, mx )
                      jobz=O   3*mn + max( 4*mn*mn + 4*mn, mx )
                      jobz=S   3*mn + max( 3*mn*mn + 4*mn, mx )
                      jobz=A   3*mn + max( 3*mn*mn + 4*mn, mx )

    @param
    iwork   (workspace) INTEGER array, dimension (8*min(M,N))

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  The updating process of DBDSDC did not converge.

    Further Details
    ---------------
    Based on contributions by
    Ming Gu and Huan Ren, Computer Science Division, University of
    California at Berkeley, USA

    @ingroup magma_gesdd
*******************************************************************************/
extern "C" magma_int_t
magma_dgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *s,
    double *U,    magma_int_t ldu,
    double *VT,   magma_int_t ldvt,
    double *work, magma_int_t lwork,
    magma_int_t *iwork,
    magma_int_t *info)
{
    dgesdd_path = "init";
    
    #define A(i_,j_)  (A  + (i_) + (j_)*lda)
    #define U(i_,j_)  (U  + (i_) + (j_)*ldu)
    #define VT(i_,j_) (VT + (i_) + (j_)*ldvt)

    // Constants
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;
    const magma_int_t izero    = 0;
    const magma_int_t ione     = 1;
    const magma_int_t ineg_one = -1;

    // Local variables
    magma_int_t lnwork, i__1;
    magma_int_t i, ie, il, ir, iu, ib;
    double dummy[1], unused[1];
    double anrm, bignum, eps, smlnum;
    magma_int_t ivt, iscl;
    magma_int_t idummy[1], ierr, itau;
    magma_int_t chunk, wrkbl, itaup, itauq;
    magma_int_t nwork;
    magma_int_t ldwrkl, ldwrkr, ldwrku, ldwrkvt, minwrk, maxwrk, mnthr;
    
    // Parameter adjustments for Fortran indexing
    A  -= 1 + lda;
    --work;

    // Function Body
    *info = 0;
    const magma_int_t m_1 = m - 1;
    const magma_int_t n_1 = n - 1;
    const magma_int_t minmn = min( m, n );
    
    const bool want_qa  = (jobz == MagmaAllVec);
    const bool want_qs  = (jobz == MagmaSomeVec);
    const bool want_qas = (want_qa || want_qs);
    const bool want_qo  = (jobz == MagmaOverwriteVec);
    const bool want_qn  = (jobz == MagmaNoVec);
    const bool lquery   = (lwork < 0);
    
    // Test the input arguments
    if (! (want_qa || want_qs || want_qo || want_qn)) {
        *info = -1;
    }
    else if (m < 0) {
        *info = -2;
    }
    else if (n < 0) {
        *info = -3;
    }
    else if (lda < max(1,m)) {
        *info = -5;
    }
    else if (ldu < 1 || (want_qas && ldu < m) || (want_qo && m < n && ldu < m)) {
        *info = -8;
    }
    else if (ldvt < 1 || (want_qa && ldvt < n) || (want_qs && ldvt < minmn)
                      || (want_qo && m >= n && ldvt < n)) {
        *info = -10;
    }
    
    //magma_int_t nb = magma_get_dgesvd_nb( m, n );
    
    // Compute workspace
    // Note: Comments in the code beginning "Workspace:" describe the
    // minimal amount of workspace needed at that point in the code,
    // as well as the preferred amount for good performance.
    // Brackets [...] indicate which matrices or vectors each term applies to.
    // NB refers to the optimal block size for the immediately
    // following subroutine, as returned by ILAENV or magma_get_*_nb.
    //
    // Comments like "geqrf = n or n*nb" indicate the minimum (n) and optimal (n*nb)
    // lwork for that LAPACK routine; MAGMA usually requires the optimal.
    //
    // Comments after maxwrk and minwrk indicate a bound using the largest NB.
    // Due to different NB for different routines, maxwrk may be less than this bound.
    // The minwrk bound is for LAPACK only;
    // MAGMA usually requires the maxwrk, and sets minwrk = maxwrk.
    //
    // wrkbl is everything except R and U (or L and VT) matrices.
    // It is used later to compute ldwrkr for R and ldwrku for U.
    // (This differs from LAPACK.)
    minwrk = 1;
    maxwrk = 1;
    wrkbl  = 1;
    mnthr = magma_int_t( minmn * 11. / 6. );
    if (*info == 0 && m > 0 && n > 0) {
        if (m >= n) {
            // Compute space needed for DBDSDC
            magma_int_t lwork_dbdsdc;
            if (want_qn) {
                // dbdsdc requires 4*n, but LAPACK has 7*n; keep 7*n for compatability.
                lwork_dbdsdc = 7*n;
            }
            else {
                lwork_dbdsdc = 3*n*n + 4*n;
            }

            // Compute space preferred for each routine
            // For MAGMA, these are all required
            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, unused, &m, unused, unused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgebrd(      m,  n, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgebrd_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dgebrd( &n, &n, unused, &n, unused, unused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgebrd(      n,  n, unused,  n, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgebrd_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dgeqrf( &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgeqrf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgeqrf_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "Q", &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaQ,   m,  n,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_q_mn = magma_int_t( real( dummy[0] ));
            
            // magma_dorgqr2 does not take workspace; use LAPACK's for compatability
            lapackf77_dorgqr( &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_dorgqr_mm = magma_int_t( real( dummy[0] ));
            
            lapackf77_dorgqr( &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_dorgqr_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "P", "R", "T",              &n, &n, &n, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaP, MagmaRight, MagmaTrans,  n,  n,  n, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_prt_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "Q", "L", "N",               &m, &m, &n, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  m,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_qln_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "Q", "L", "N",               &m, &n, &n, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  n,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_qln_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "Q", "L", "N",               &n, &n, &n, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans,  n,  n,  n, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_qln_nn = magma_int_t( real( dummy[0] ));
            
            if (m >= mnthr) {
                if (want_qn) {
                    // Path 1 (M >> N, JOBZ='N')
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn     );   // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsdc        );   // bdsdc  = 7*n (or 4*n)
                    maxwrk = wrkbl;                                     // maxwrk = 3*n + max(5*n, 2*n*nb)
                    //                                              lapack minwrk = 8*n
                }
                else if (want_qo) {
                    // Path 2 (M >> N, JOBZ='O')
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn     );   // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn     );   // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_nn );   // ormbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    // Technically, only bdsdc and ormbr need U  matrix,
                    // but accounting for that only changes maxwrk for n < 2*nb/3, and LAPACK doesn't
                    // todo: is m*n needed, or is n*n enough? lapack uses n*n in dgesdd, m*n in zgesdd.
                    maxwrk = n*n + n*n + wrkbl;                         // maxwrk = 2*n*n + 3*n + max(3*n*n + 4*n, 2*n*nb)
                    minwrk = n*n + n*n + wrkbl;                         // minwrk = 2*n*n + 3*n + max(3*n*n + 4*n, 2*n*nb)
                    //                                              lapack minwrk = 5*n*n + 7*n
                }
                else if (want_qs) {
                    // Path 3 (M >> N, JOBZ='S')
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn     );   // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn     );   // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_nn );   // ormbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 3*n + max(3*n*n + 4*n, 2*n*nb)
                    //                                              lapack minwrk = 4*n*n + 7*n
                }
                else if (want_qa) {
                    // Path 4 (M >> N, JOBZ='A')
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn     );   // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mm     );   // orgqr  = m or m*nb (note m); magma_dorgqr2 doesn't need work
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_nn );   // ormbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(3*n*n + 7*n, n + m*nb, 3*n + 2*n*nb); magma doesn't need n + m*nb
                    //                                              lapack minwrk = n*n + max(3*n*n + 7*n, m + n) [fixed]
                    //                                              lapack minwrk = n*n + 2*n + m + (3*n*n + 4*n) [original]
                }
            }
            else {
                // Path 5 (M >= N, but not much larger)
                wrkbl      = max( wrkbl, 3*n + lwork_dgebrd_mn     );   // gebrd  = m or (m+n)*nb (note m)
                if (want_qn) {
                    // Path 5n (M >= N, JOBZ='N')
                    // dgebrd above      3*n + lwork_dgebrd_mn          // gebrd  = m or (m+n)*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 7*n (or 4*n)
                    maxwrk = wrkbl;                                     // maxwrk = 3*n + max(7*n, (m+n)*nb)
                    //                                              lapack minwrk = 3*n + max(7*n, m)
                }
                else if (want_qo) {
                    // Path 5o (M >= N, JOBZ='O')
                    // dgebrd above      3*n + lwork_dgebrd_mn          // gebrd  = m or (m+n)*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    
                    // Path 5o-fast
                    // Uses m*n for U,  no R matrix, and ormbr.
                    // Technically, only bdsdc and ormbr need U  matrix,
                    // but accounting for that only changes maxwrk for n < nb, and LAPACK doesn't
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_mn );   // ormbr  = n or n*nb
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 3*n + max(3*n*n + 4*n, (m+n)*nb)
                    
                    // Path 5o-slow
                    // Uses n*n for U,  lwork=nb*n for R in gemm, and orgbr.
                    minwrk = max( wrkbl, 3*n + lwork_dorgbr_q_mn );     // orgbr  = n or n*nb
                    minwrk = n*n + minwrk;                              // minwrk = n*n + 3*n + max(3*n*n + 4*n, (m+n)*nb)
                    //                                              lapack minwrk = 3*n + max(4*n*n + 4*n, m)
                }
                else if (want_qs) {
                    // Path 5s (M >= N, JOBZ='S')
                    // dgebrd above      3*n + lwork_dgebrd_mn          // gebrd  = m or (m+n)*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_mn );   // ormbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 3*n + max(3*n*n + 4*n, (m+n)*nb)
                    //                                              lapack minwrk = 3*n + max(3*n*n + 4*n, m)
                }
                else if (want_qa) {
                    // Path 5a (M >= N, JOBZ='A')
                    // dgebrd above      3*n + lwork_dgebrd_mn          // gebrd  = m or (m+n)*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dbdsdc        );   // bdsdc  = 3*n*n + 4*n
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 3*n + max(3*n*n + 4*n, (m+n)*nb)
                    //                                              lapack minwrk = 3*n + max(3*n*n + 4*n, m)
                }
            }
        }
        else {
            // m < n
            // Compute space needed for DBDSDC
            magma_int_t lwork_dbdsdc;
            if (want_qn) {
                // dbdsdc requires 4*m, but LAPACK has 7*m; keep 7*m for compatability.
                lwork_dbdsdc = 7*m;
            }
            else {
                lwork_dbdsdc = 3*m*m + 4*m;
            }
            
            // Compute space preferred for each routine
            // For MAGMA, these are all required
            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, unused, &m, unused, unused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgebrd(      m,  n, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgebrd_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dgebrd( &m, &m, unused, &m, unused, unused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgebrd(      m,  m, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgebrd_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dgelqf( &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dgelqf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dgelqf_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "P", &m, &n, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaP,   m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_p_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorglq( &m, &n, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorglq(      m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorglq_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorglq( &n, &n, &m, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorglq(      n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorglq_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "P", "R", "T",              &m, &m, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaP, MagmaRight, MagmaTrans,  m,  m,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_prt_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "P", "R", "T",              &m, &n, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaP, MagmaRight, MagmaTrans,  m,  n,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_prt_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "P", "R", "T",               &n, &n, &m, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n,  n,  m, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_prt_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dormbr( "Q", "L", "N",               &m, &m, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  m,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_qln_mm = magma_int_t( real( dummy[0] ));
            
            if (n >= mnthr) {
                if (want_qn) {
                    // Path 1t (N >> M, JOBZ='N')
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn     );   // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsdc        );   // bdsdc  = 7*m (or 4*m)
                    maxwrk = wrkbl;                                     // maxwrk = 3*m + max(5*m, 2*m*nb)
                    //                                              lapack minwrk = 8*m
                }
                else if (want_qo) {
                    // Path 2t (N >> M, JOBZ='O')
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn     );   // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn     );   // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_mm );   // ormbr  = m or m*nb
                    // Technically, only bdsdc and ormbr need VT matrix,
                    // but accounting for that only changes maxwrk for m < 2*nb/3, and LAPACK doesn't
                    // todo: is m*n needed, or is m*m enough? lapack uses m*m in dgesdd, m*n in zgesdd.
                    maxwrk = m*m + m*m + wrkbl;                         // maxwrk = 2*m*m + 3*m + max(3*m*m + 4*m, 2*m*nb)
                    minwrk = m*m + m*m + wrkbl;                         // minwrk = 2*m*m + 3*m + max(3*m*m + 4*m, 2*m*nb)
                    //                                              lapack minwrk = 5*m*m + 7*m
                }
                else if (want_qs) {
                    // Path 3t (N >> M, JOBZ='S')
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn     );   // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn     );   // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_mm );   // ormbr  = m or m*nb
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 3*m + max(3*m*m + 4*m, 2*m*nb)
                    //                                              lapack minwrk = 4*m*m + 7*m
                }
                else if (want_qa) {
                    // Path 4t (N >> M, JOBZ='A')
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn     );   // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_nn     );   // orglq  = n or n*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_mm );   // ormbr  = m or m*nb
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(3*m*m + 7*m, m + n*nb, 3*m + 2*m*nb)
                    //                                              lapack minwrk = m*m + max(3*m*m + 7*m, m + n) [fixed]
                    //                                              lapack minwrk = m*m + 2*m + n + (3*m*m + 4*m) [original]
                }
            }
            else {
                // Path 5t (N > M, but not much larger)
                wrkbl      = max( wrkbl, 3*m + lwork_dgebrd_mn     );   // gebrd  = n or (m+n)*nb (note n)
                if (want_qn) {
                    // Path 5tn (N > M, JOBZ='N')
                    // dgebrd above      3*m + lwork_dgebrd_mn          // gebrd  = n or (m+n)*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 7*m (or 4*m)
                    maxwrk = wrkbl;                                     // maxwrk = 3*m + max(7*m, (m+n)*nb)
                    //                                              lapack minwrk = 3*m + max(7*m, n)
                }
                else if (want_qo) {
                    // Path 5to (N > M, JOBZ='O')
                    // dgebrd above      3*m + lwork_dgebrd_mn          // gebrd  = n or (m+n)*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    
                    // Path 5to-fast
                    // Uses n*m for VT, no L matrix, and ormbr.
                    // Technically, only bdsdc and ormbr need VT matrix,
                    // but accounting for that only changes maxwrk for m < nb, and LAPACK doesn't
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_mn );   // ormbr  = m or m*nb
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 3*m + max(3*m*m + 4*m, (m+n)*nb)
                    
                    // Path 5to-slow
                    // Uses m*m for VT, lwork=nb*m for L in gemm, and orgbr.
                    minwrk = max( wrkbl, 3*m + lwork_dorgbr_p_mn );     // orgbr  = m or m*nb
                    minwrk = m*m + minwrk;                              // minwrk = m*m + 3*m + max(3*m*m + 4*m, (m+n)*nb)
                    //                                              lapack minwrk = 3*m + max(4*m*m + 4*m, n)
                }
                else if (want_qs) {
                    // Path 5ts (N > M, JOBZ='S')
                    // dgebrd above      3*m + lwork_dgebrd_mn          // gebrd  = n or (m+n)*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_mn );   // ormbr  = m or m*nb
                    maxwrk = wrkbl;                                     // maxwrk = 3*m + max(3*m*m + 4*m, (m+n)*nb)
                    //                                              lapack minwrk = 3*m + max(3*m*m + 4*m, n)
                }
                else if (want_qa) {
                    // Path 5ta (N > M, JOBZ='A')
                    // dgebrd above      3*m + lwork_dgebrd_mn          // gebrd  = n or (m+n)*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dbdsdc        );   // bdsdc  = 3*m*m + 4*m
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_qln_mm );   // ormbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dormbr_prt_nn );   // ormbr  = n or n*nb (note n)
                    maxwrk = wrkbl;                                     // maxwrk = 3*m + max(3*m*m + 4*m, (m+n)*nb)
                    //                                              lapack minwrk = 3*m + max(3*m*m + 4*m, n)
                }
            }
        }
        // unlike lapack, magma usually requires maxwrk, unless minwrk was set above
        if (minwrk == 1) {
            minwrk = maxwrk;
        }
        maxwrk = max( maxwrk, minwrk );
        
        work[1] = magma_dmake_lwork( maxwrk );

        if (lwork < minwrk && ! lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return *info;
    }

    // Get machine constants
    eps = lapackf77_dlamch("P");
    smlnum = sqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;

    // Scale A if max element outside range [SMLNUM,BIGNUM]
    anrm = lapackf77_dlange( "M", &m, &n, A(1,1), &lda, dummy );
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_dlascl( "G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr );
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_dlascl( "G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr );
    }

    if (m >= n) {                                                 //
        // A has at least as many rows as columns.
        // If A has sufficiently more rows than columns, first reduce using
        // the QR decomposition (if sufficient workspace available)
        if (m >= mnthr) {                                         //
            if (want_qn) {                                        //
                // Path 1 (M >> N, JOBZ='N')
                dgesdd_path = "1n";
                // No singular vectors to be computed
                itau  = 1;
                nwork = itau + n;

                // Compute A=Q*R
                // Workspace: need   N [tau] + N    [geqrf work]
                // Workspace: prefer N [tau] + N*NB [geqrf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgeqrf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Zero out below R
                lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda );
                ie    = 1;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in A
                // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &n, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      n,  n, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif
                nwork = ie + n;

                // Perform bidiagonal SVD, computing singular values only
                // Workspace: need   N [e] + 4*N [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &n, s, &work[ie], dummy, &ione, dummy, &ione, dummy, idummy, &work[nwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 2 (M >> N, JOBZ='O')
                dgesdd_path = "2o";
                // N left  singular vectors to be overwritten on A and
                // N right singular vectors to be computed in VT

                // WORK[IR] is LDWRKR by N, at least N*N, up to M*N
                // after accounting for geqrf, gebrd, bdsdc, etc. workspace,
                // make gemm workspace (R) as tall as possible, from N up to M.
                ldwrkr = min( lda, (lwork - n*n - wrkbl)/n );
                assert( ldwrkr >= n );
                ir    = 1;
                itau  = ir + ldwrkr*n;
                nwork = itau + n;

                // Compute A=Q*R
                // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgeqrf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Copy R to WORK[IR], zeroing out below it
                lapackf77_dlacpy( "U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr );
                lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr );

                // Generate Q in A
                // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
                // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorgqr( &m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorgqr2(     m,  n,  n, A(1,1),  lda, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;
                
                // Bidiagonalize R in WORK[IR]
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // WORK[IU] is N by N
                iu    = nwork;
                nwork = iu + n*n;

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in WORK[IU] and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], &work[iu], &n, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite WORK[IU] by left  singular vectors of R, and
                // overwrite VT       by right singular vectors of R
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N    [ormbr work]
                // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &n,    &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT,        &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], &work[iu], n,    &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, n, &work[ir], ldwrkr, &work[itaup], VT,        ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in A by left singular vectors of R in WORK[IU],
                // storing result in WORK[IR] and copying to A
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U]
                // Workspace: prefer M*N [R] + 3*N [e, tauq, taup] + N*N [U]
                for (i = 1; i <= m; i += ldwrkr) {
                    ib = min( m - i + 1, ldwrkr );
                    blasf77_dgemm( "N", "N", &ib, &n, &n, &c_one, A(i,1), &lda, &work[iu], &n, &c_zero, &work[ir], &ldwrkr );
                    lapackf77_dlacpy( "F", &ib, &n, &work[ir], &ldwrkr, A(i,1), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 3 (M >> N, JOBZ='S')
                dgesdd_path = "3s";
                // N left  singular vectors to be computed in U and
                // N right singular vectors to be computed in VT
                ir = 1;

                // WORK[IR] is N by N
                ldwrkr = n;
                itau  = ir + ldwrkr*n;
                nwork = itau + n;

                // Compute A=Q*R
                // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgeqrf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Copy R to WORK[IR], zeroing out below it
                lapackf77_dlacpy( "U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr );
                lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr );

                // Generate Q in A
                // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
                // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorgqr( &m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorgqr2(     m,  n,  n, A(1,1),  lda, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in WORK[IR]
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U  by left  singular vectors of R, and
                // overwrite VT by right singular vectors of R
                // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [ormbr work]
                // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in A by left singular vectors of R in WORK[IR],
                // storing result in U
                // Workspace: need   N*N [R]
                lapackf77_dlacpy( "F", &n, &n, U, &ldu, &work[ir], &ldwrkr );
                blasf77_dgemm( "N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U, &ldu );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 4 (M >> N, JOBZ='A')
                dgesdd_path = "4a";
                // M left  singular vectors to be computed in U and
                // N right singular vectors to be computed in VT

                // WORK[IU] is N by N
                ldwrku = n;
                iu    = 1;
                itau  = iu + ldwrku*n;
                nwork = itau + n;

                // Compute A=Q*R, copying result to U
                // Workspace: need   N*N [U] + N [tau] + N    [geqrf work]
                // Workspace: prefer N*N [U] + N [tau] + N*NB [geqrf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgeqrf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif
                lapackf77_dlacpy( "L", &m, &n, A(1,1), &lda, U, &ldu );

                // Generate Q in U
                // Workspace: need   N*N [U] + N [tau] + M    [orgqr work]
                // Workspace: prefer N*N [U] + N [tau] + M*NB [orgqr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorgqr2( m, m, n, U, ldu, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif

                // Produce R in A, zeroing out other entries
                lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda );
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in A
                // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [gebrd work]
                // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &n, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      n,  n, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in WORK[IU] and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], &work[iu], &n, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite WORK[IU] by left  singular vectors of R, and
                // overwrite VT       by right singular vectors of R
                // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [ormbr work]
                // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT,        &ldvt,   &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, n, A(1,1), lda, &work[itaup], VT,        ldvt,   &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in U by left singular vectors of R in WORK[IU],
                // storing result in A
                // Workspace: need   N*N [U]
                blasf77_dgemm( "N", "N", &m, &n, &n, &c_one, U, &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda );

                // Copy left singular vectors of A from A to U
                lapackf77_dlacpy( "F", &m, &n, A(1,1), &lda, U, &ldu );
            }                                                     //
        }                                                         //
        else {                                                    //
            // M < MNTHR
            // Path 5 (M >= N, but not much larger)
            dgesdd_path = "5";
            // Reduce to bidiagonal form without QR decomposition
            ie    = 1;
            itauq = ie    + n;
            itaup = itauq + n;
            nwork = itaup + n;

            // Bidiagonalize A
            // Workspace: need   3*N [e, tauq, taup] + M        [gebrd work]
            // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [gebrd work]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_dgebrd(      m,  n, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
            #endif
            
            if (want_qn) {                                        //
                // Path 5n (M >= N, JOBZ='N')
                dgesdd_path = "5n";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace: need   3*N [e, tauq, taup] + 4*N [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &n, s, &work[ie], dummy, &ione, dummy, &ione, dummy, idummy, &work[nwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 5o (M >= N, JOBZ='O')
                dgesdd_path = "5o";
                iu = nwork;
                if (lwork >= m*n + wrkbl) {
                    // WORK[IU] is M by N
                    // WORK[IR] is not used in this case
                    ldwrku = m;
                    nwork = iu + ldwrku*n;
                    lapackf77_dlaset( "F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku );
                    ir = -1;  // unused
                }
                else {
                    // WORK[IU] is N by N
                    ldwrku = n;
                    nwork = iu + ldwrku*n;

                    // WORK[IR] is LDWRKR by N
                    ir = nwork;
                    ldwrkr = min( m, (lwork - n*n - 3*n) / n );
                }

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in WORK[IU] and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   3*N [e, tauq, taup] + N*N [U] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], &work[iu], &ldwrku, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite VT by right singular vectors of A
                // Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [ormbr work]
                // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                if (lwork >= m*n + wrkbl) {
                    // Path 5o-fast
                    dgesdd_path = "5o-fast";
                    // Overwrite WORK[IU] by left singular vectors of A
                    // Workspace: need   3*N [e, tauq, taup] + M*N [U] + N    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*N [U] + N*NB [ormbr work]
                    assert( ldwrku == m );
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr );
                    #endif
                
                    // Copy left singular vectors of A from WORK[IU] to A
                    lapackf77_dlacpy( "F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda );
                }
                else {
                    // Path 5o-slow
                    dgesdd_path = "5o-slow";
                    // Generate Q in A
                    // Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [orgbr work]
                    lnwork = lwork - nwork + 1;
                    lapackf77_dorgbr( "Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr );

                    // Multiply Q in A by left singular vectors of
                    // bidiagonal matrix in WORK[IU], storing result in
                    // WORK[IR] and copying to A
                    // Workspace: need   3*N [e, tauq, taup] + N*N [U] + NB*N [R]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + M*N  [R]
                    for (i = 1; i <= m; i += ldwrkr) {
                        ib = min( m - i + 1, ldwrkr );
                        blasf77_dgemm( "N", "N", &ib, &n, &n, &c_one, A(i,1), &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr );
                        lapackf77_dlacpy( "F", &ib, &n, &work[ir], &ldwrkr, A(i,1), &lda );
                    }
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 5s (M >= N, JOBZ='S')
                dgesdd_path = "5s";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   3*N [e, tauq, taup] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dlaset( "F", &m, &n, &c_zero, &c_zero, U, &ldu );
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U  by left  singular vectors of A, and
                // overwrite VT by right singular vectors of A
                // Workspace: need   3*N [e, tauq, taup] + N    [ormbr work]
                // Workspace: prefer 3*N [e, tauq, taup] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
            else if (want_qa) {                                   //
                // Path 5a (M >= N, JOBZ='A')
                dgesdd_path = "5a";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   3*N [e, tauq, taup] + (3*N*N + 4*N) [bdsdc work]
                lapackf77_dlaset( "F", &m, &m, &c_zero, &c_zero, U, &ldu );
                lapackf77_dbdsdc( "U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Set the right corner of U to identity matrix
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_dlaset( "F", &i__1, &i__1, &c_zero, &c_one, U(n,n), &ldu );
                }

                // Overwrite U  by left  singular vectors of A, and
                // overwrite VT by right singular vectors of A
                // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
        }                                                         //
    }                                                             //
    else {                                                        //
        // A has more columns than rows.
        // If A has sufficiently more columns than rows, first reduce using
        // the LQ decomposition (if sufficient workspace available)
        if (n >= mnthr) {                                         //
            if (want_qn) {                                        //
                // Path 1t (N >> M, JOBZ='N')
                dgesdd_path = "1tn";
                // No singular vectors to be computed
                itau  = 1;
                nwork = itau + m;

                // Compute A=L*Q
                // Workspace: need   M [tau] + M    [gelqf work]
                // Workspace: prefer M [tau] + M*NB [gelqf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgelqf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Zero out above L
                lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda );
                ie    = 1;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in A
                // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &m, &m, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      m,  m, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif
                nwork = ie + m;

                // Perform bidiagonal SVD, computing singular values only
                // Workspace: need   M [e] + 4*M [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &m, s, &work[ie], dummy, &ione, dummy, &ione, dummy, idummy, &work[nwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 2t (N >> M, JOBZ='O')
                dgesdd_path = "2to";
                // M right singular vectors to be overwritten on A and
                // M left  singular vectors to be computed in U

                // WORK[IVT] is M by M
                // WORK[IL]  is M by M; it is later resized to M by chunk for gemm
                // Path 2 puts R first and U after tauq.
                // That could be done here, putting L first and VT after tauq;
                // doing so would change chunk (subtract wrkbl) and itau (use ldwrkl*chunk).
                // See dgesdd Path 2t.
                ldwrkl = m;
                ivt   = 1;
                il    = ivt + m*m;
                itau  = il + ldwrkl*m;
                nwork = itau + m;

                // Compute A=L*Q
                // Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [gelqf work]
                // Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [gelqf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgelqf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Copy L to WORK[IL], zeroing out above it
                lapackf77_dlacpy( "L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl );
                lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl );

                // Generate Q in A
                // Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [orglq work]
                // Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [orglq work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorglq( &m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorglq(      m,  n,  m, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in WORK[IL]
                // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M      [gebrd work]
                // Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &m, &m, &work[il], &ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      m,  m, &work[il],  ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U, and
                // computing right singular vectors of bidiagonal matrix in WORK[IVT]
                // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &m, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U         by left  singular vectors of L, and
                // overwrite WORK[IVT] by right singular vectors of L
                // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M    [ormbr work]
                // Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U,          &ldu, &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &m,   &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U,          ldu, &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   m, m, m, &work[il], ldwrkl, &work[itaup], &work[ivt], m,   &work[nwork], lnwork, &ierr );
                #endif
                
                // resize L to M x chunk, at least M*M, up to M*N
                chunk = min( n, (lwork - m*m)/m );
                assert( chunk >= m );
                
                // Multiply right singular vectors of L in WORK[IVT] by Q in A,
                // storing result in WORK[IL] and copying to A
                // Workspace: need   M*M [VT] + M*M [L]
                // Workspace: prefer M*M [VT] + M*N [L]
                for (i = 1; i <= n; i += chunk) {
                    ib = min( n - i + 1, chunk );
                    blasf77_dgemm( "N", "N", &m, &ib, &m, &c_one, &work[ivt], &m, A(1,i), &lda, &c_zero, &work[il], &ldwrkl );
                    lapackf77_dlacpy( "F", &m, &ib, &work[il], &ldwrkl, A(1,i), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 3t (N >> M, JOBZ='S')
                dgesdd_path = "3ts";
                // M right singular vectors to be computed in VT and
                // M left  singular vectors to be computed in U
                il = 1;

                // WORK[IL] is M by M
                ldwrkl = m;
                itau  = il + ldwrkl*m;
                nwork = itau + m;

                // Compute A=L*Q
                // Workspace: need   M*M [L] + M [tau] + M    [gelqf work]
                // Workspace: prefer M*M [L] + M [tau] + M*NB [gelqf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgelqf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Copy L to WORK[IL], zeroing out above it
                lapackf77_dlacpy( "L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl );
                lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl );

                // Generate Q in A
                // Workspace: need   M*M [L] + M [tau] + M    [orglq work]
                // Workspace: prefer M*M [L] + M [tau] + M*NB [orglq work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorglq( &m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorglq(      m,  n,  m, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in WORK[IU]
                // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M      [gebrd work]
                // Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &m, &m, &work[il], &ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      m,  m, &work[il],  ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U  by left  singular vectors of L, and
                // overwrite VT by right singular vectors of L
                // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M    [ormbr work]
                // Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   m, m, m, &work[il], ldwrkl, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply right singular vectors of L in WORK[IL] by Q in A,
                // storing result in VT
                // Workspace: need   M*M [L]
                lapackf77_dlacpy( "F", &m, &m, VT, &ldvt, &work[il], &ldwrkl );
                blasf77_dgemm( "N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT, &ldvt );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 4t (N >> M, JOBZ='A')
                dgesdd_path = "4ta";
                // N right singular vectors to be computed in VT and
                // M left  singular vectors to be computed in U

                // WORK[IVT] is M by M
                ldwrkvt = m;
                ivt   = 1;
                itau  = ivt + ldwrkvt*m;
                nwork = itau + m;

                // Compute A=L*Q, copying result to VT
                // Workspace: need   M*M [VT] + M [tau] + M    [gelqf work]
                // Workspace: prefer M*M [VT] + M [tau] + M*NB [gelqf work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgelqf(      m,  n, A(1,1),  lda, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif
                lapackf77_dlacpy( "U", &m, &n, A(1,1), &lda, VT, &ldvt );

                // Generate Q in VT
                // Workspace: need   M*M [VT] + M [tau] + N    [orglq work]
                // Workspace: prefer M*M [VT] + M [tau] + N*NB [orglq work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[nwork],  lnwork, &ierr );
                #endif

                // Produce L in A, zeroing out other entries
                lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda );
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in A
                // Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M      [gebrd work]
                // Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &m, &m, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_dgebrd(      m,  m, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in WORK[IVT]
                // Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dbdsdc( "U", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &ldwrkvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U         by left  singular vectors of L, and
                // overwrite WORK[IVT] by right singular vectors of L
                // Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M    [ormbr work]
                // Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U,          &ldu,     &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwrkvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, A(1,1), lda, &work[itauq], U,          ldu,     &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   m, m, m, A(1,1), lda, &work[itaup], &work[ivt], ldwrkvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply right singular vectors of L in WORK[IVT] by Q in VT,
                // storing result in A
                // Workspace: need   M*M [VT]
                blasf77_dgemm( "N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwrkvt, VT, &ldvt, &c_zero, A(1,1), &lda );

                // Copy right singular vectors of A from A to VT
                lapackf77_dlacpy( "F", &m, &n, A(1,1), &lda, VT, &ldvt );
            }                                                     //
        }                                                         //
        else {                                                    //
            // N < MNTHR
            // Path 5t (N > M, but not much larger)
            dgesdd_path = "5t";
            // Reduce to bidiagonal form without LQ decomposition
            ie    = 1;
            itauq = ie    + m;
            itaup = itauq + m;
            nwork = itaup + m;

            // Bidiagonalize A
            // Workspace: need   3*M [e, tauq, taup] + N        [gebrd work]
            // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [gebrd work]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_dgebrd(      m,  n, A(1,1),  lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork],  lnwork, &ierr );
            #endif
            
            if (want_qn) {                                        //
                // Path 5tn (N > M, JOBZ='N')
                dgesdd_path = "5tn";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace: need   3*M [e, tauq, taup] + 4*M [bdsdc work]
                lapackf77_dbdsdc( "L", "N", &m, s, &work[ie], dummy, &ione, dummy, &ione, dummy, idummy, &work[nwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 5to (N > M, JOBZ='O')
                dgesdd_path = "5to";
                ldwrkvt = m;
                ivt = nwork;
                if (lwork >= m*n + wrkbl) {
                    // WORK[IVT] is M by N
                    // WORK[IL] is not used in this case
                    lapackf77_dlaset( "F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwrkvt );
                    nwork = ivt + ldwrkvt*n;
                    il    = -1;  // unused
                    chunk = -1;  // unused
                }
                else {
                    // WORK[IVT] is M by M
                    nwork = ivt + ldwrkvt*m;
                    il = nwork;

                    // WORK[IL] is M by CHUNK
                    chunk = min( n, (lwork - m*m - 3*m) / m );
                }

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in WORK[IVT]
                // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dbdsdc( "L", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &ldwrkvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U by left singular vectors of A
                // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [ormbr work]
                // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                if (lwork >= m*n + wrkbl) {
                    // Path 5to-fast
                    dgesdd_path = "5to-fast";
                    // Overwrite WORK[IVT] by left singular vectors of A
                    // Workspace: need   3*M [e, tauq, taup] + M*N [VT] + M    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*N [VT] + M*NB [ormbr work]
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwrkvt, &work[nwork], &lnwork, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaRight, MagmaTrans, m, n, m, A(1,1), lda, &work[itaup], &work[ivt], ldwrkvt, &work[nwork], lnwork, &ierr );
                    #endif
                
                    // Copy right singular vectors of A from WORK[IVT] to A
                    lapackf77_dlacpy( "F", &m, &n, &work[ivt], &ldwrkvt, A(1,1), &lda );
                }
                else {
                    // Path 5to-slow
                    dgesdd_path = "5to-slow";
                    // Generate P**T in A
                    // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [orgbr work]
                    lnwork = lwork - nwork + 1;
                    lapackf77_dorgbr( "P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr );

                    // Multiply Q in A by right singular vectors of
                    // bidiagonal matrix in WORK[IVT], storing result in
                    // WORK[IL] and copying to A
                    // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M*NB [L]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*N  [L]
                    for (i = 1; i <= n; i += chunk) {
                        ib = min( n - i + 1, chunk );
                        blasf77_dgemm( "N", "N", &m, &ib, &m, &c_one, &work[ivt], &ldwrkvt, A(1,i), &lda, &c_zero, &work[il], &m );
                        lapackf77_dlacpy( "F", &m, &ib, &work[il], &m, A(1,i), &lda );
                    }
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 5ts (N > M, JOBZ='S')
                dgesdd_path = "5ts";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   3*M [e, tauq, taup] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dlaset( "F", &m, &n, &c_zero, &c_zero, VT, &ldvt );
                lapackf77_dbdsdc( "L", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Overwrite U  by left  singular vectors of A, and
                // overwrite VT by right singular vectors of A
                // Workspace: need   3*M [e, tauq, taup] + M    [ormbr work]
                // Workspace: prefer 3*M [e, tauq, taup] + M*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   m, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
            else if (want_qa) {                                   //
                // Path 5ta (N > M, JOBZ='A')
                dgesdd_path = "5ta";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in U and
                // computing right singular vectors of bidiagonal matrix in VT
                // Workspace: need   3*M [e, tauq, taup] + (3*M*M + 4*M) [bdsdc work]
                lapackf77_dlaset( "F", &n, &n, &c_zero, &c_zero, VT, &ldvt );
                lapackf77_dbdsdc( "L", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dummy, idummy, &work[nwork], iwork, info );

                // Set the right corner of VT to identity matrix
                if (n > m) {
                    i__1 = n - m;
                    lapackf77_dlaset( "F", &i__1, &i__1, &c_zero, &c_one, VT(m,m), &ldvt );
                }

                // Overwrite U  by left  singular vectors of A, and
                // overwrite VT by right singular vectors of A
                // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr );
                lapackf77_dormbr( "P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_dormbr( MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr );
                magma_dormbr( MagmaP, MagmaRight, MagmaTrans,   n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
        }                                                         //
    }                                                             //

    // Undo scaling if necessary
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl( "G", &izero, &izero, &bignum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
        if (anrm < smlnum) {
            lapackf77_dlascl( "G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
    }

    // Return optimal workspace in WORK[1] (Fortran index)
    work[1] = magma_dmake_lwork( maxwrk );
    
    return *info;
} // magma_dgesdd
