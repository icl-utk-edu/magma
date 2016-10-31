/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> c

*/
#include "magma_internal.h"

#define COMPLEX

// Version 1 - LAPACK
// Version 2 - MAGMA
#define VERSION 2

const char* zgesdd_path = "none";

/***************************************************************************//**
    Purpose
    -------
    ZGESDD computes the singular value decomposition (SVD) of a complex
    M-by-N matrix A, optionally computing the left and right singular
    vectors, by using divide-and-conquer method. The SVD is written

        A = U * SIGMA * conjugate-transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
    V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**H, not V.

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
      -     = MagmaAllVec:  all M columns of U and all N rows of V**H are
                            returned in the arrays U and VT;
      -     = MagmaSomeVec: the first min(M,N) columns of U and
                            the first min(M,N) rows of V**H are
                            returned in the arrays U and VT;
      -     = MagmaOverwriteVec:
                    If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**H are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**H are overwritten
                    on the array A;
      -     = MagmaNoVec:   no columns of U or rows of V**H are computed.

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
      -     if JOBZ = MagmaOverwriteVec,
                if M >= N, A is overwritten with the first N columns
                of U (the left singular vectors, stored columnwise);
                otherwise, A is overwritten with the first M rows
                of V**H (the right singular vectors, stored rowwise).
      -     if JOBZ != MagmaOverwriteVec, the contents of A are destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       COMPLEX_16 array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N;
            UCOL = min(M,N) if JOBZ = MagmaSomeVec.
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N,
            U contains the M-by-M unitary matrix U;
      -     if JOBZ = MagmaSomeVec, U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBZ = MagmaOverwriteVec and M >= N, or JOBZ = MagmaNoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = MagmaSomeVec or MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N, LDU >= M.

    @param[out]
    VT      COMPLEX_16 array, dimension (LDVT,N)
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N,
            VT contains the N-by-N unitary matrix V**H;
      -     if JOBZ = MagmaSomeVec, VT contains the first min(M,N) rows of
            V**H (the right singular vectors, stored rowwise);
      -     if JOBZ = MagmaOverwriteVec and M < N, or JOBZ = MagmaNoVec, VT is not referenced.

    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N, LDVT >= N;
            if JOBZ = MagmaSomeVec, LDVT >= min(M,N).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,lwork))
            On exit, if INFO = 0, WORK[0] returns the optimal lwork.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= mn*17/9.
            For job: N=None, O=Overwrite, S=Some, A=All.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_zgesvd_nb(M,N).
    \n
            Optimal lwork (required in MAGMA)
            for mx >> mn:
            Path 1:   jobz=N                  2*mn + 2*mn*nb
            Path 2:   jobz=O  2*mn*mn       + 2*mn + 2*mn*nb
                          or  mx*mn + mn*mn + 2*mn + 2*mn*nb  [marginally faster?]
            Path 3:   jobz=S  mn*mn         + 2*mn + 2*mn*nb
            Path 4:   jobz=A  mn*mn +    max( 2*mn + 2*mn*nb, mn + mx*nb )
            for mx >= mn, but not mx >> mn:
            Path 5,6: jobz=N          2*mn + (mx + mn)*nb
                      jobz=O  mx*mn + 2*mn + (mx + mn)*nb  [faster algorithm]
                          or  mn*mn + 2*mn + (mx + mn)*nb  [slower algorithm]
                      jobz=S          2*mn + (mx + mn)*nb
                      jobz=A          2*mn + (mx + mn)*nb
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork
            for mx >> mn:
            Path 1:   jobz=N            3*mn
            Path 2:   jobz=O  2*mn*mn + 3*mn
            Path 3:   jobz=S    mn*mn + 3*mn
            Path 4:   jobz=A    mn*mn + 2*mn + mx          # LAPACK's overestimate
                         or     mn*mn + max( m + n, 3*n )  # correct minimum
            for mx >= mn, but not mx >> mn:
            Path 5,6: jobz=N            2*mn + mx
                      jobz=O    mn*mn + 2*mn + mx
                      jobz=S            2*mn + mx
                      jobz=A            2*mn + mx

    @param
    rwork   (workspace) DOUBLE PRECISION array, dimension (MAX(1,LRWORK))
            Let mx = max(M,N) and mn = min(M,N).
            These sizes should work for both MAGMA and LAPACK.
            If JOBZ =  MagmaNoVec, LRWORK >= 5*mn.  # LAPACK <= 3.6 had bug requiring 7*mn
            If JOBZ != MagmaNoVec,
                if mx >> mn,       LRWORK >=      5*mn*mn + 5*mn;
                otherwise,         LRWORK >= max( 5*mn*mn + 5*mn,
                                                  2*mx*mn + 2*mn*mn + mn ).
    \n
            For JOBZ = MagmaNoVec, some implementations seem to have a bug requiring
            LRWORK >= 7*mn in some cases.

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
magma_zgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *s,
    magmaDoubleComplex *U, magma_int_t ldu,
    magmaDoubleComplex *VT, magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info)
{
    zgesdd_path = "init";
    
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    #define U(i_,j_) (U + (i_) + (j_)*ldu)
    #define VT(i_,j_) (VT + (i_) + (j_)*ldvt)

    // Constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    const magma_int_t izero  = 0;
    const magma_int_t ione   = 1;
    const magma_int_t ineg_one = -1;

    // Local variables
    magma_int_t lnwork, i__1;
    magma_int_t i, ie, il, ir, iu, ib;
    magmaDoubleComplex dummy[1], unused[1];
    double rdummy[1], runused[1];
    double anrm, bignum, eps, smlnum;
    magma_int_t ivt, iscl;
    magma_int_t idummy[1], ierr, itau;
    magma_int_t chunk, wrkbl, itaup, itauq;
    magma_int_t nwork;
    magma_int_t ldwrkl, ldwrkr, ldwrku, ldwrkvt, minwrk, maxwrk, mnthr1, mnthr2;
    magma_int_t iru, irvt, nrwork;
    
    // Parameter adjustments for Fortran indexing
    A  -= 1 + lda;
    --work;
    --rwork;

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

    //magma_int_t nb = magma_get_zgesvd_nb( m, n );

    // Compute workspace
    // Note: Comments in the code beginning "Workspace:" describe the
    // minimal amount of workspace needed at that point in the code,
    // as well as the preferred amount for good performance.
    // Workspace refers to complex workspace, and RWorkspace to real workspace.
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
    // Note: rwork in path 5, JOBZ='O' depends on mnthr1 < 2 * minmn.
    mnthr1 = magma_int_t( minmn * 17. / 9. );
    mnthr2 = magma_int_t( minmn *  5. / 3. );
    if (*info == 0) {
        if (m >= n && minmn > 0) {
            // There is no complex work space needed for bidiagonal SVD (dbdsdc);
            // the real work space (LRWORK) it needs is listed in the documentation above.

            // Compute space preferred for each routine
            // For MAGMA, these are all required
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, unused, &m, runused, runused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgebrd(      m,  n, unused,  m, runused, runused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgebrd_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zgebrd( &n, &n, unused, &n, runused, runused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgebrd(      n,  n, unused,  n, runused, runused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgebrd_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zgeqrf( &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgeqrf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgeqrf_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "P", &n, &n, &n, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaP,   n,  n,  n, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_p_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "Q", &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaQ,   m,  m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_q_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "Q", &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaQ,   m,  n,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_q_mn = magma_int_t( real( dummy[0] ));
            
            // magma_zungqr2 does not take workspace; use LAPACK's for compatability
            lapackf77_zungqr( &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_zungqr_mm = magma_int_t( real( dummy[0] ));
            
            lapackf77_zungqr( &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_zungqr_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "P", "R", "C",                  &n, &n, &n, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans,  n,  n,  n, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_prc_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "Q", "L", "N",               &m, &m, &n, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  m,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_qln_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "Q", "L", "N",               &m, &n, &n, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  n,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_qln_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "Q", "L", "N",               &n, &n, &n, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans,  n,  n,  n, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_qln_nn = magma_int_t( real( dummy[0] ));
            
            if (m >= mnthr1) {
                if (want_qn) {
                    // Path 1 (M >> N, JOBZ='N')
                    wrkbl = max( wrkbl,   n + lwork_zgeqrf_mn );        // geqrf  = n or   n*nb
                    wrkbl = max( wrkbl, 2*n + lwork_zgebrd_nn );        // gebrd  = n or 2*n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + 2*n*nb
                    //                                              lapack minwrk = 3*n
                }
                else if (want_qo) {
                    // Path 2 (M >> N, JOBZ='O')
                    wrkbl  = max( wrkbl,   n + lwork_zgeqrf_mn     );   // geqrf  = n or   n*nb
                    wrkbl  = max( wrkbl,   n + lwork_zungqr_mn     );   // ungqr  = n or   n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_qln_nn );   // unmbr  = n or   n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );   // unmbr  = n or   n*nb
                    // todo: is m*n needed, or is n*n enough?
                    maxwrk = m*n + n*n + wrkbl;                         // maxwrk = m*n + n*n + 2*n + 2*n*nb
                    minwrk = n*n + n*n + wrkbl;                         // minwrk = 2*n*n     + 2*n + 2*n*nb
                    //                                              lapack minwrk = 2*n*n + 3*n
                }
                else if (want_qs) {
                    // Path 3 (M >> N, JOBZ='S')
                    wrkbl  = max( wrkbl,   n + lwork_zgeqrf_mn     );   // geqrf  = n or   n*nb
                    wrkbl  = max( wrkbl,   n + lwork_zungqr_mn     );   // ungqr  = n or   n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_qln_nn );   // unmbr  = n or   n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );   // unmbr  = n or   n*nb
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 2*n + 2*n*nb
                    //                                              lapack minwrk = n*n + 3*n
                }
                else if (want_qa) {
                    // Path 4 (M >> N, JOBZ='A')
                    wrkbl  = max( wrkbl,   n + lwork_zgeqrf_mn     );   // geqrf  = n or   n*nb
                    wrkbl  = max( wrkbl,   n + lwork_zungqr_mm     );   // ungqr  = m or   m*nb (note m)
                    wrkbl  = max( wrkbl, 2*n + lwork_zgebrd_nn     );   // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_qln_nn );   // unmbr  = n or   n*nb
                    wrkbl  = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );   // unmbr  = n or   n*nb
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(n + m*nb, 2*n + 2*n*nb)
                    //                                              lapack minwrk = n*n + max(m + n, 3*n) [fixed]
                    //                                              lapack minwrk = n*n + 2*n + m         [original]
                }
            }
            else if (m >= mnthr2) {
                // Path 5 (M >> N, but not as much as MNTHR1)
                wrkbl     = max( wrkbl, 2*n + lwork_zgebrd_mn );        // gebrd  = m or (m+n)*nb (note m)
                if (want_qn) {
                    // Path 5n (M >> N, JOBZ='N')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
                else if (want_qo) {
                    // Path 5o (M >> N, JOBZ='O')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_p_nn );      // ungbr  = n or n*nb
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_q_mn );      // ungbr  = n or n*nb
                    // todo: is m*n needed, or is n*n enough?
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 2*n + (m+n)*nb
                    minwrk = n*n + wrkbl;                               // minwrk = n*n + 2*n + (m+n)*nb
                    //                                              lapack minwrk = n*n + 2*n + m
                }
                else if (want_qs) {
                    // Path 5s (M >> N, JOBZ='S')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_p_nn );      // ungbr  = n or n*nb
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_q_mn );      // ungbr  = n or n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
                else if (want_qa) {
                    // Path 5a (M >> N, JOBZ='A')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_p_nn );      // ungbr  = n or n*nb
                    wrkbl = max( wrkbl, 2*n + lwork_zungbr_q_mm );      // ungbr  = m or m*nb (note m)
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
            }
            else {
                // Path 6 (M >= N, but not much larger)
                wrkbl     = max( wrkbl, 2*n + lwork_zgebrd_mn );        // gebrd  = m or (m+n)*nb (note m)
                if (want_qn) {
                    // Path 6n (M >= N, JOBZ='N')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
                else if (want_qo) {
                    // Path 6o (M >= N, JOBZ='O')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );    // unmbr  = n or n*nb
                    
                    // Path 6o-fast
                    // Uses m*n for U,  no R matrix, and unmbr.
                    // Technically, gebrd doesn't need U  matrix,
                    // but accounting for that only changes maxwrk for n < nb
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_qln_mn );    // unmbr  = n or n*nb
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 2*n + (m+n)*nb
                    
                    // Path 6o-slow
                    // Uses n*n for U,  lwork=nb*n for R in gemm, and ungbr.
                    minwrk = max( wrkbl, 2*n + lwork_zungbr_q_mn );     // ungbr  = n or n*nb
                    minwrk = n*n + minwrk;                              // minwrk = n*n + 2*n + (m+n)*nb
                    //                                              lapack minwrk = n*n + 2*n + m
                }
                else if (want_qs) {
                    // Path 6s (M >= N, JOBZ='S')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_qln_mn );    // unmbr  = n or n*nb
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );    // unmbr  = n or n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
                else if (want_qa) {
                    // Path 6a (M >= N, JOBZ='A')
                    // zgebrd above     2*n + lwork_zgebrd_mn           // gebrd  = m or (m+n)*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_qln_mm );    // unmbr  = m or m*nb (note m)
                    wrkbl = max( wrkbl, 2*n + lwork_zunmbr_prc_nn );    // unmbr  = n or n*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*n + (m+n)*nb
                    //                                              lapack minwrk = 2*n + m
                }
            }
        }
        else if (minmn > 0) {
            // m < n
            // Compute space preferred for each routine
            // For MAGMA, these are all required
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, unused, &m, runused, runused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgebrd(      m,  n, unused,  m, runused, runused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgebrd_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zgebrd( &m, &m, unused, &m, runused, runused, unused, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgebrd(      m,  m, unused,  m, runused, runused, unused, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgebrd_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zgelqf( &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zgelqf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zgelqf_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "P", &m, &n, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaP,   m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_p_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "P", &n, &n, &m, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaP,   n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_p_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zungbr( "Q", &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zungbr( MagmaQ,   m,  m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zungbr_q_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunglq( &m, &n, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zunglq(      m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunglq_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunglq( &n, &n, &m, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_zunglq(      n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunglq_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "P", "R", "C",                  &m, &m, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans,  m,  m,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_prc_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "P", "R", "C",                  &m, &n, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans,  m,  n,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_prc_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "P", "R", "C",                  &n, &n, &m, unused, &n, unused, unused, &n, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans,  n,  n,  m, unused,  n, unused, unused,  n, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_prc_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_zunmbr( "Q", "L", "N",               &m, &m, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans,  m,  m,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_zunmbr_qln_mm = magma_int_t( real( dummy[0] ));
            
            if (n >= mnthr1) {
                if (want_qn) {
                    // Path 1t (N >> M, JOBZ='N')
                    wrkbl = max( wrkbl,   m + lwork_zgelqf_mn );        // gelqf  = m or   m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zgebrd_mm );        // gebrd  = m or 2*m*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + 2*m*nb
                    //                                              lapack minwrk = 3*m
                }
                else if (want_qo) {
                    // Path 2t (N >> M, JOBZ='O')
                    wrkbl  = max( wrkbl,   m + lwork_zgelqf_mn     );   // gelqf  = m or   m*nb
                    wrkbl  = max( wrkbl,   m + lwork_zunglq_mn     );   // unglq  = m or   m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );   // unmbr  = m or   m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_prc_mm );   // unmbr  = m or   m*nb
                    // todo: is m*n needed, or is m*m enough?
                    maxwrk = m*n + m*m + wrkbl;                         // maxwrk = m*n + m*m + 2*m + 2*m*nb
                    minwrk = m*m + m*m + wrkbl;                         // minwrk = 2*m*m     + 2*m + 2*m*nb
                    //                                              lapack minwrk = 2*m*m + 3*m
                }
                else if (want_qs) {
                    // Path 3t (N >> M, JOBZ='S')
                    wrkbl  = max( wrkbl,   m + lwork_zgelqf_mn     );   // gelqf  = m or   m*nb
                    wrkbl  = max( wrkbl,   m + lwork_zunglq_mn     );   // unglq  = m or   m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );   // unmbr  = m or   m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_prc_mm );   // unmbr  = m or   m*nb
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 2*m + 2*m*nb
                    //                                              lapack minwrk = m*m + 3*m
                }
                else if (want_qa) {
                    // Path 4t (N >> M, JOBZ='A')
                    wrkbl  = max( wrkbl,   m + lwork_zgelqf_mn     );   // gelqf  = m or   m*nb
                    wrkbl  = max( wrkbl,   m + lwork_zunglq_nn     );   // unglq  = n or   n*nb (note n)
                    wrkbl  = max( wrkbl, 2*m + lwork_zgebrd_mm     );   // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );   // unmbr  = m or   m*nb
                    wrkbl  = max( wrkbl, 2*m + lwork_zunmbr_prc_mm );   // unmbr  = m or   m*nb
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(m + n*nb, 2*m + 2*m*nb)
                    //                                              lapack minwrk = m*m + max(m + n, 3*m) [fixed]
                    //                                              lapack minwrk = m*m + 2*m + n         [original]
                }
            }
            else if (n >= mnthr2) {
                // Path 5t (N >> M, but not as much as MNTHR1)
                wrkbl     = max( wrkbl, 2*m + lwork_zgebrd_mn );        // gebrd  = n or (m+n)*nb (note n)
                if (want_qn) {
                    // Path 5tn (N >> M, JOBZ='N')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
                else if (want_qo) {
                    // Path 5to (N >> M, JOBZ='O')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_q_mm );      // ungbr  = m or m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_p_mn );      // ungbr  = m or m*nb
                    // todo: is m*n needed, or is m*m enough?
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 2*m + (m+n)*nb
                    minwrk = m*m + wrkbl;                               // minwrk = m*m + 2*m + (m+n)*nb
                    //                                              lapack minwrk = m*m + 2*m + n
                }
                else if (want_qs) {
                    // Path 5ts (N >> M, JOBZ='S')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_q_mm );      // ungbr  = m or m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_p_mn );      // ungbr  = m or m*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
                else if (want_qa) {
                    // Path 5ta (N >> M, JOBZ='A')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_q_mm );      // ungbr  = m or m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zungbr_p_nn );      // ungbr  = n or n*nb (note n)
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
            }
            else {
                // Path 6t (N > M, but not much larger)
                wrkbl     = max( wrkbl, 2*m + lwork_zgebrd_mn );        // gebrd  = n or (m+n)*nb (note n)
                if (want_qn) {
                    // Path 6tn (N > M, JOBZ='N')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
                else if (want_qo) {
                    // Path 6to (N > M, JOBZ='O')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );    // unmbr  = m or m*nb
                    
                    // Path 6to-fast
                    // Uses m*n for VT, no L matrix, and unmbr.
                    // Technically, gebrd doesn't need VT matrix,
                    // but accounting for that only changes maxwrk for m < nb
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_prc_mn );    // unmbr  = m or m*nb
                    maxwrk = m*n + wrkbl;                               // maxwrk = m*n + 2*m + (m+n)*nb
                    
                    // Path 6to-slow
                    // Uses m*m for VT, lwork=nb*m for L in gemm, and ungbr.
                    minwrk = max( wrkbl, 2*m + lwork_zungbr_p_mn );     // ungbr  = m or m*nb
                    minwrk = m*m + minwrk;                              // minwrk = m*m + 2*m + (m+n)*nb
                    //                                              lapack minwrk = m*m + 2*m + n
                }
                else if (want_qs) {
                    // Path 6ts (N > M, JOBZ='S')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );    // unmbr  = m or m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_prc_mn );    // unmbr  = m or m*nb
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
                else if (want_qa) {
                    // Path 6ta (N > M, JOBZ='A')
                    // zgebrd above     2*m + lwork_zgebrd_mn           // gebrd  = n or (m+n)*nb (note n)
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_qln_mm );    // unmbr  = m or m*nb
                    wrkbl = max( wrkbl, 2*m + lwork_zunmbr_prc_nn );    // unmbr  = n or n*nb (note n)
                    maxwrk = wrkbl;                                     // maxwrk = 2*m + (m+n)*nb
                    //                                              lapack minwrk = 2*m + n
                }
            }
        }
        // unlike lapack, magma usually requires maxwrk, unless minwrk was set above
        if (minwrk == 1) {
            minwrk = maxwrk;
        }
        maxwrk = max( maxwrk, minwrk );
        
        work[1] = magma_zmake_lwork( maxwrk );
        
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
    if (m  == 0 || n == 0) {
        return *info;
    }

    // Get machine constants
    eps = lapackf77_dlamch("P");
    smlnum = sqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;

    // Scale A if max element outside range [SMLNUM, BIGNUM]
    anrm = lapackf77_zlange( "M", &m, &n, A(1,1), &lda, rdummy );
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_zlascl( "G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr );
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_zlascl( "G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr );
    }

    if (m >= n) {                                                 //
        // A has at least as many rows as columns.
        // If A has sufficiently more rows than columns, first reduce using
        // the QR decomposition (if sufficient workspace available)
        if (m >= mnthr1) {                                        //
            if (want_qn) {                                        //
                // Path 1 (M >> N, JOBZ='N')
                zgesdd_path = "1n";
                // No singular vectors to be computed
                itau  = 1;
                nwork = itau + n;
                
                // Compute A=Q*R
                // Workspace:  need   N [tau] + N    [geqrf work]
                // Workspace:  prefer N [tau] + N*NB [geqrf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgeqrf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                
                // Zero out below R
                lapackf77_zlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda );
                ie    = 1;
                itauq = 1;
                itaup = itauq + n;
                nwork = itaup + n;
                
                // Bidiagonalize R in A
                // Workspace:  need   2*N [tauq, taup] + N      [gebrd work]
                // Workspace:  prefer 2*N [tauq, taup] + 2*N*NB [gebrd work]
                // RWorkspace: need   N [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &n, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( n, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif
                nrwork = ie + n;
                
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   N [e] + 4*N [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &n, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 2 (M >> N, JOBZ='O')
                zgesdd_path = "2o";
                // N left  singular vectors to be overwritten on A and
                // N right singular vectors to be computed in VT
                iu = 1;

                // WORK[IU] is N by N
                ldwrku = n;
                ir     = iu + ldwrku*n;
                // weird: was m*n + n*n + 3*n, which means lapack prefers
                // having m*n R to having n*nb for other routines -- seems dumb.
                if (lwork >= m*n + n*n + wrkbl) {
                    // WORK[IR] is M by N
                    // replace one N*N with M*N in comments denoted ## below
                    ldwrkr = m;
                }
                else {
                    // WORK[IR] is N by N
                    ldwrkr = (lwork - n*n - wrkbl) / n;
                    assert( ldwrkr >= n );
                }
                itau  = ir + ldwrkr*n;
                nwork = itau + n;

                // Compute A=Q*R
                // Workspace:  need   N*N [U] + N*N [R] + N [tau] + N    [geqrf work]
                // Workspace:  prefer N*N [U] + N*N [R] + N [tau] + N*NB [geqrf work] ##
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgeqrf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif

                // Copy R to WORK[ IR ], zeroing out below it
                lapackf77_zlacpy( "U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr );
                lapackf77_zlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr );

                // Generate Q in A
                // Workspace:  need   N*N [U] + N*N [R] + N [tau] + N    [ungqr work]
                // Workspace:  prefer N*N [U] + N*N [R] + N [tau] + N*NB [ungqr work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungqr( &m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungqr2( m, n, n, A(1,1), lda, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif
                ie    = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in WORK[IR]
                // Workspace:  need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N      [gebrd work]
                // Workspace:  prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + 2*N*NB [gebrd work] ##
                // RWorkspace: need   N [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &n, &n, &work[ir], &ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( n, n, &work[ir], ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif
                
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix WORK[IU]
                // Overwrite WORK[IU] by the left singular vectors of R
                // Workspace:  need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [unmbr work] ##
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by the right singular vectors of R
                // Workspace:  need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [unmbr work] ##
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in A by left singular vectors of R in WORK[IU],
                // storing result in WORK[IR] and copying to A
                // Workspace:  need   N*N [U] + N*N [R]
                // Workspace:  prefer N*N [U] + M*N [R]
                // RWorkspace: need   0
                for (i = 1; i <= m; i += ldwrkr) {
                    ib = min( m - i + 1, ldwrkr );
                    blasf77_zgemm( "N", "N", &ib, &n, &n, &c_one, A(i,1), &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr );
                    lapackf77_zlacpy( "F", &ib, &n, &work[ir], &ldwrkr, A(i,1), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 3 (M >> N, JOBZ='S')
                zgesdd_path = "3s";
                // N left  singular vectors to be computed in U and
                // N right singular vectors to be computed in VT
                ir = 1;

                // WORK[IR] is N by N
                ldwrkr = n;
                itau   = ir + ldwrkr*n;
                nwork  = itau + n;

                // Compute A=Q*R
                // Workspace:  need   N*N [R] + N [tau] + N    [geqrf work]
                // Workspace:  prefer N*N [R] + N [tau] + N*NB [geqrf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgeqrf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif

                // Copy R to WORK[IR], zeroing out below it
                lapackf77_zlacpy( "U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr );
                lapackf77_zlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr );

                // Generate Q in A
                // Workspace:  need   N*N [R] + N [tau] + N    [ungqr work]
                // Workspace:  prefer N*N [R] + N [tau] + N*NB [ungqr work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungqr( &m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungqr2( m, n, n, A(1,1), lda, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif
                ie    = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in WORK[IR]
                // Workspace:  need   N*N [R] + 2*N [tauq, taup] + N      [gebrd work]
                // Workspace:  prefer N*N [R] + 2*N [tauq, taup] + 2*N*NB [gebrd work]
                // RWorkspace: need   N [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &n, &n, &work[ir], &ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( n, n, &work[ir], ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of R
                // Workspace:  need   N*N [R] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [R] + 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of R
                // Workspace:  need   N*N [R] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [R] + 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in A by left singular vectors of R in WORK[IR],
                // storing result in U
                // Workspace:  need   N*N [R]
                // RWorkspace: need   0
                lapackf77_zlacpy( "F", &n, &n, U, &ldu, &work[ir], &ldwrkr );
                blasf77_zgemm( "N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U, &ldu );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 4 (M >> N, JOBZ='A')
                zgesdd_path = "4a";
                // M left  singular vectors to be computed in U and
                // N right singular vectors to be computed in VT
                iu = 1;

                // WORK[IU] is N by N
                ldwrku = n;
                itau   = iu + ldwrku*n;
                nwork  = itau + n;

                // Compute A=Q*R, copying result to U
                // Workspace:  need   N*N [U] + N [tau] + N    [geqrf work]
                // Workspace:  prefer N*N [U] + N [tau] + N*NB [geqrf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgeqrf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgeqrf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                lapackf77_zlacpy( "L", &m, &n, A(1,1), &lda, U, &ldu );

                // Generate Q in U
                // Workspace:  need   N*N [U] + N [tau] + M    [ungqr work]
                // Workspace:  prefer N*N [U] + N [tau] + M*NB [ungqr work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungqr( &m, &m, &n, U, &ldu, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungqr2( m, m, n, U, ldu, &work[itau], /*&work[nwork], lnwork,*/ &ierr );
                #endif

                // Produce R in A, zeroing out below it
                lapackf77_zlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda );
                ie    = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                // Bidiagonalize R in A
                // Workspace:  need   N*N [U] + 2*N [tauq, taup] + N      [gebrd work]
                // Workspace:  prefer N*N [U] + 2*N [tauq, taup] + 2*N*NB [gebrd work]
                // RWorkspace: need   N [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &n, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( n, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix WORK[IU]
                // Overwrite WORK[IU] by left singular vectors of R
                // Workspace:  need   N*N [U] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [U] + 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of R
                // Workspace:  need   N*N [U] + 2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer N*N [U] + 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply Q in U by left singular vectors of R in WORK[IU],
                // storing result in A
                // Workspace:  need   N*N [U]
                // RWorkspace: need   0
                blasf77_zgemm( "N", "N", &m, &n, &n, &c_one, U, &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda );

                // Copy left singular vectors of A from A to U
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, U, &ldu );
            }                                                     //
        }                                                         //
        else if (m >= mnthr2) {                                   //
            // MNTHR2 <= M < MNTHR1
            // Path 5 (M >> N, but not as much as MNTHR1)
            zgesdd_path = "5";
            // Reduce to bidiagonal form without QR decomposition, use
            // ZUNGBR and matrix multiplication to compute singular vectors
            ie     = 1;
            nrwork = ie + n;
            itauq  = 1;
            itaup  = itauq + n;
            nwork  = itaup + n;

            // Bidiagonalize A
            // Workspace:  need   2*N [tauq, taup] + M        [gebrd work]
            // Workspace:  prefer 2*N [tauq, taup] + (M+N)*NB [gebrd work]
            // RWorkspace: need   N [e]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_zgebrd( m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
            #endif
            
            if (want_qn) {                                        //
                // Path 5n (M >> N, JOBZ='N')
                zgesdd_path = "5n";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   N [e] + 4*N [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &n, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 5o (M >> N, JOBZ='O')
                zgesdd_path = "5o";

                // Copy A to VT, generate P**H
                // Workspace:  need   2*N [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "U", &n, &n, A(1,1), &lda, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, n, n, n, VT, ldvt, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Generate Q in A
                // Workspace:  need   2*N [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, n, n, A(1,1), lda, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                iu = nwork;
                if (lwork >= m*n + wrkbl) {
                    // WORK[IU] is M by N
                    ldwrku = m;
                }
                else {
                    // WORK[IU] is LDWRKU by N
                    ldwrku = (lwork - wrkbl) / n;
                    assert( ldwrku >= n );
                }
                nwork = iu + ldwrku*n;

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply real matrix RWORK[IRVT] by P**H in VT,
                // storing the result in WORK[IU], copying to VT
                // Workspace:  need   2*N [tauq, taup] + N*N [U]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [larcm work]
                lapackf77_zlarcm( &n, &n, &rwork[irvt], &n, VT, &ldvt, &work[iu], &ldwrku, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &n, &n, &work[iu], &ldwrku, VT, &ldvt );

                // Multiply Q in A by real matrix RWORK[IRU],
                // storing the result in WORK[IU], copying to A
                // Workspace:  need   2*N [tauq, taup] + N*N [U]
                // Workspace:  prefer 2*N [tauq, taup] + M*N [U]
                // RWorkspace: need   N [e] + N*N [RU] + 2*N*N [lacrm work]
                // RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [lacrm work] < N + 5*N*N since M < 2*N here
                nrwork = irvt;
                for (i = 1; i <= m; i += ldwrku) {
                    ib = min( m - i + 1, ldwrku );
                    lapackf77_zlacrm( &ib, &n, A(i,1), &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork] );
                    lapackf77_zlacpy( "F", &ib, &n, &work[iu], &ldwrku, A(i,1), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 5s (M >> N, JOBZ='S')
                zgesdd_path = "5s";
                // Copy A to VT, generate P**H
                // Workspace:  need   2*N [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "U", &n, &n, A(1,1), &lda, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, n, n, n, VT, ldvt, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Copy A to U, generate Q
                // Workspace:  need   2*N [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "L", &m, &n, A(1,1), &lda, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &n, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, n, n, U, ldu, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply real matrix RWORK[IRVT] by P**H in VT,
                // storing the result in A, copying to VT
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [larcm work]
                lapackf77_zlarcm( &n, &n, &rwork[irvt], &n, VT, &ldvt, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &n, &n, A(1,1), &lda, VT, &ldvt );

                // Multiply Q in U by real matrix RWORK[IRU],
                // storing the result in A, copying to U
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + 2*M*N [lacrm work] < N + 5*N*N since M < 2*N here
                nrwork = irvt;
                lapackf77_zlacrm( &m, &n, U, &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, U, &ldu );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 5a (M >> N, JOBZ='A')
                zgesdd_path = "5a";
                // Copy A to VT, generate P**H
                // Workspace:  need   2*N [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "U", &n, &n, A(1,1), &lda, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, n, n, n, VT, ldvt, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Copy A to U, generate Q
                // Workspace:  need   2*N [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*N [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "L", &m, &n, A(1,1), &lda, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, m, n, U, ldu, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply real matrix RWORK[IRVT] by P**H in VT,
                // storing the result in A, copying to VT
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [larcm work]
                lapackf77_zlarcm( &n, &n, &rwork[irvt], &n, VT, &ldvt, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &n, &n, A(1,1), &lda, VT, &ldvt );

                // Multiply Q in U by real matrix RWORK[IRU],
                // storing the result in A, copying to U
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + 2*M*N [lacrm work] < N + 5*N*N since M < 2*N here
                nrwork = irvt;
                lapackf77_zlacrm( &m, &n, U, &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, U, &ldu );
            }                                                     //
        }                                                         //
        else {                                                    //
            // M < MNTHR2
            // Path 6 (M >= N, but not much larger)
            zgesdd_path = "6";
            // Reduce to bidiagonal form without QR decomposition
            // Use ZUNMBR to compute singular vectors
            ie     = 1;
            nrwork = ie + n;
            itauq  = 1;
            itaup  = itauq + n;
            nwork  = itaup + n;

            // Bidiagonalize A
            // Workspace:  need   2*N [tauq, taup] + M        [gebrd work]
            // Workspace:  prefer 2*N [tauq, taup] + (M+N)*NB [gebrd work]
            // RWorkspace: need   N [e]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_zgebrd( m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
            #endif
            
            if (want_qn) {                                        //
                // Path 6n (M >= N, JOBZ='N')
                zgesdd_path = "6n";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   N [e] + 4*N [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &n, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 6o (M >= N, JOBZ='O')
                zgesdd_path = "6o";
                iu = nwork;
                if (lwork >= m*n + wrkbl) {
                    // WORK[IU] is M by N
                    ldwrku = m;
                }
                else {
                    // WORK[IU] is LDWRKU by N
                    ldwrku = (lwork - wrkbl) / n;
                }
                nwork = iu + ldwrku*n;

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of A
                // Workspace:  need   2*N [tauq, taup] + N*N [U] + N    [unmbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*N [U] + N*NB [unmbr work]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                if (lwork >= m*n + wrkbl) {
                    // Path 6o-fast
                    zgesdd_path = "6o-fast";
                    // Copy real matrix RWORK[IRU] to complex matrix WORK[IU]
                    // Overwrite WORK[IU] by left singular vectors of A, copying to A
                    // Workspace:  need   2*N [tauq, taup] + M*N [U] + N    [unmbr work]
                    // Workspace:  prefer 2*N [tauq, taup] + M*N [U] + N*NB [unmbr work]
                    // RWorkspace: need   N [e] + N*N [RU]
                    lapackf77_zlaset( "F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku );
                    lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku );
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_zunmbr( "Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr );
                    #else
                    magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr );
                    #endif
                    lapackf77_zlacpy( "F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda );
                }
                else {
                    // Path 6o-slow
                    zgesdd_path = "6o-slow";
                    // Generate Q in A
                    // Workspace:  need   2*N [tauq, taup] + N*N [U] + N    [ungbr work]
                    // Workspace:  prefer 2*N [tauq, taup] + N*N [U] + N*NB [ungbr work]
                    // RWorkspace: need   0
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_zungbr( "Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr );
                    #else
                    magma_zungbr( MagmaQ, m, n, n, A(1,1), lda, &work[itauq], &work[nwork], lnwork, &ierr );
                    #endif

                    // Multiply Q in A by real matrix RWORK[IRU],
                    // storing the result in WORK[IU], copying to A
                    // Workspace:  need   2*N [tauq, taup] + N*N [U]
                    // Workspace:  prefer 2*N [tauq, taup] + M*N [U]
                    // RWorkspace: need   N [e] + N*N [RU] + 2*N*N [lacrm work]
                    // RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [lacrm work] < N + 5*N*N since M < 2*N here
                    nrwork = irvt;
                    for (i = 1; i <= m; i += ldwrku) {
                        ib = min( m - i + 1, ldwrku );
                        lapackf77_zlacrm( &ib, &n, A(i,1), &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork] );
                        lapackf77_zlacpy( "F", &ib, &n, &work[iu], &ldwrku, A(i,1), &lda );
                    }
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 6s (M >= N, JOBZ='S')
                zgesdd_path = "6s";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of A
                // Workspace:  need   2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
                lapackf77_zlaset( "F", &m, &n, &c_zero, &c_zero, U, &ldu );
                lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of A
                // Workspace:  need   2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
            else if (want_qa) {                                   //
                // Path 6a (M >= N, JOBZ='A')
                zgesdd_path = "6a";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + (3*N*N + 4*N) [bdsdc work]
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc( "U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Set the right corner of U to identity matrix
                lapackf77_zlaset( "F", &m, &m, &c_zero, &c_zero, U, &ldu );
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_zlaset( "F", &i__1, &i__1, &c_zero, &c_one, U(n,n), &ldu );
                }

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of A
                // Workspace:  need   2*N [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer 2*N [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
                lapackf77_zlacp2( "F", &n, &n, &rwork[iru], &n, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of A
                // Workspace:  need   2*N [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer 2*N [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
                lapackf77_zlacp2( "F", &n, &n, &rwork[irvt], &n, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
        }                                                         //
    }                                                             //
    else {                                                        //
        // m < n
        // A has more columns than rows.
        // If A has sufficiently more columns than rows, first reduce using
        // the LQ decomposition (if sufficient workspace available)
        if (n >= mnthr1) {                                        //
            if (want_qn) {                                        //
                // Path 1t (N >> M, JOBZ='N')
                zgesdd_path = "1tn";
                // No singular vectors to be computed
                itau  = 1;
                nwork = itau + m;
                
                // Compute A=L*Q
                // Workspace:  need   M [tau] + M    [gelqf work]
                // Workspace:  prefer M [tau] + M*NB [gelqf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgelqf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                
                // Zero out above L
                lapackf77_zlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda );
                ie    = 1;
                itauq = 1;
                itaup = itauq + m;
                nwork = itaup + m;
                
                // Bidiagonalize L in A
                // Workspace:  need   2*M [tauq, taup] + M      [gebrd work]
                // Workspace:  prefer 2*M [tauq, taup] + 2*M*NB [gebrd work]
                // RWorkspace: need   M [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &m, &m, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( m, m, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif
                nrwork = ie + m;
                
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   M [e] + 4*M [bdsdc work]
                lapackf77_dbdsdc( "U", "N", &m, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 2t (N >> M, JOBZ='O')
                zgesdd_path = "2to";
                // M right singular vectors to be overwritten on A and
                // M left  singular vectors to be computed in U
                ivt = 1;

                // WORK[IVT] is M by M
                ldwrkvt = m;
                il = ivt + ldwrkvt*m;
                // todo: lapack has 3*m instead of wrkbl; prefers m*n L to having m*nb?
                if (lwork >= m*n + m*m + wrkbl) {
                    // WORK[IL] is M by N
                    // replace one M*M with M*N in comments denoted ## above
                    ldwrkl = m;
                    chunk  = n;
                }
                else {
                    // WORK[IL] is M by CHUNK
                    ldwrkl = m;
                    chunk  = (lwork - m*m - wrkbl) / m;
                }
                itau  = il + ldwrkl*chunk;
                nwork = itau + m;

                // Compute A=L*Q
                // Workspace:  need   M*M [VT] + M*M [L] + M [tau] + M    [gelqf work]
                // Workspace:  prefer M*M [VT] + M*M [L] + M [tau] + M*NB [gelqf work] ##
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgelqf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif

                // Copy L to WORK[IL], zeroing out above it
                lapackf77_zlacpy( "L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl );
                lapackf77_zlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl );

                // Generate Q in A
                // Workspace:  need   M*M [VT] + M*M [L] + M [tau] + M    [unglq work]
                // Workspace:  prefer M*M [VT] + M*M [L] + M [tau] + M*NB [unglq work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunglq( &m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zunglq( m, n, m, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                ie    = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in WORK[IL]
                // Workspace:  need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M      [gebrd work]
                // Workspace:  prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + 2*M*NB [gebrd work] ##
                // RWorkspace: need   M [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &m, &m, &work[il], &ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( m, m, &work[il], ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + (3*M*M + 4*M) [bdsdc work]
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc( "U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix WORK[IU]
                // Overwrite WORK[IU] by the left singular vectors of L
                // Workspace:  need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [unmbr work] ##
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT]
                // Overwrite WORK[IVT] by the right singular vectors of L
                // Workspace:  need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB [unmbr work] ##
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwrkvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &ldwrkvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, m, m, m, &work[il], ldwrkl, &work[itaup], &work[ivt], ldwrkvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply right singular vectors of L in WORK[IL] by Q in A,
                // storing result in WORK[IL] and copying to A
                // Workspace:  need   M*M [VT] + M*M [L]
                // Workspace:  prefer M*M [VT] + M*N [L]
                // RWorkspace: need   0
                for (i = 1; i <= n; i += chunk) {
                    ib = min( n - i + 1, chunk );
                    blasf77_zgemm( "N", "N", &m, &ib, &m, &c_one, &work[ivt], &m, A(1,i), &lda, &c_zero, &work[il], &ldwrkl );
                    lapackf77_zlacpy( "F", &m, &ib, &work[il], &ldwrkl, A(1,i), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 3t (N >> M, JOBZ='S')
                zgesdd_path = "3ts";
                // M right singular vectors to be computed in VT and
                // M left  singular vectors to be computed in U
                il = 1;

                // WORK[IL] is M by M
                ldwrkl = m;
                itau   = il + ldwrkl*m;
                nwork  = itau + m;

                // Compute A=L*Q
                // Workspace:  need   M*M [L] + M [tau] + M    [gelqf work]
                // Workspace:  prefer M*M [L] + M [tau] + M*NB [gelqf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgelqf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif

                // Copy L to WORK[IL], zeroing out above it
                lapackf77_zlacpy( "L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl );
                lapackf77_zlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl );

                // Generate Q in A
                // Workspace:  need   M*M [L] + M [tau] + M    [unglq work]
                // Workspace:  prefer M*M [L] + M [tau] + M*NB [unglq work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunglq( &m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zunglq( m, n, m, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                ie    = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in WORK[IL]
                // Workspace:  need   M*M [L] + 2*M [tauq, taup] + M      [gebrd work]
                // Workspace:  prefer M*M [L] + 2*M [tauq, taup] + 2*M*NB [gebrd work]
                // RWorkspace: need   M [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &m, &m, &work[il], &ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( m, m, &work[il], ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + (3*M*M + 4*M) [bdsdc work]
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc( "U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of L
                // Workspace:  need   M*M [L] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [L] + 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by left singular vectors of L
                // Workspace:  need   M*M [L] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [L] + 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, m, m, m, &work[il], ldwrkl, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif

                // Copy VT to WORK[IL], multiply right singular vectors of L
                // in WORK[IL] by Q in A, storing result in VT
                // Workspace:  need   M*M [L]
                // RWorkspace: need   0
                lapackf77_zlacpy( "F", &m, &m, VT, &ldvt, &work[il], &ldwrkl );
                blasf77_zgemm( "N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT, &ldvt );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 4t (N >> M, JOBZ='A')
                zgesdd_path = "4ta";
                // N right singular vectors to be computed in VT and
                // M left  singular vectors to be computed in U
                ivt = 1;

                // WORK[IVT] is M by M
                ldwrkvt = m;
                itau  = ivt + ldwrkvt*m;
                nwork = itau + m;

                // Compute A=L*Q, copying result to VT
                // Workspace:  need   M*M [VT] + M [tau] + M    [gelqf work]
                // Workspace:  prefer M*M [VT] + M [tau] + M*NB [gelqf work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgelqf( &m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgelqf( m, n, A(1,1), lda, &work[itau], &work[nwork], lnwork, &ierr );
                #endif
                lapackf77_zlacpy( "U", &m, &n, A(1,1), &lda, VT, &ldvt );

                // Generate Q in VT
                // Workspace:  need   M*M [VT] + M [tau] + N    [unglq work]
                // Workspace:  prefer M*M [VT] + M [tau] + N*NB [unglq work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[nwork], &lnwork, &ierr );
                #else
                magma_zunglq( n, n, m, VT, ldvt, &work[itau], &work[nwork], lnwork, &ierr );
                #endif

                // Produce L in A, zeroing out above it
                lapackf77_zlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda );
                ie    = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                // Bidiagonalize L in A
                // Workspace:  need   M*M [VT] + 2*M [tauq, taup] + M      [gebrd work]
                // Workspace:  prefer M*M [VT] + 2*M [tauq, taup] + 2*M*NB [gebrd work]
                // RWorkspace: need   M [e]
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zgebrd( &m, &m, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zgebrd( m, m, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + (3*M*M + 4*M) [bdsdc work]
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc( "U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of L
                // Workspace:  need   M*M [VT] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [VT] + 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT]
                // Overwrite WORK[IVT] by right singular vectors of L
                // Workspace:  need   M*M [VT] + 2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer M*M [VT] + 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   0
                lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwrkvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwrkvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, m, m, m, A(1,1), lda, &work[itaup], &work[ivt], ldwrkvt, &work[nwork], lnwork, &ierr );
                #endif

                // Multiply right singular vectors of L in WORK[IVT] by Q in VT,
                // storing result in A
                // Workspace:  need   M*M [VT]
                // RWorkspace: need   0
                blasf77_zgemm( "N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwrkvt, VT, &ldvt, &c_zero, A(1,1), &lda );

                // Copy right singular vectors of A from A to VT
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, VT, &ldvt );
            }                                                     //
        }                                                         //
        else if (n >= mnthr2) {                                   //
            // MNTHR2 <= N < MNTHR1
            // Path 5t (N >> M, but not as much as MNTHR1)
            zgesdd_path = "5t";
            // Reduce to bidiagonal form without LQ decomposition, use
            // ZUNGBR and matrix multiplication to compute singular vectors
            ie     = 1;
            nrwork = ie + m;
            itauq  = 1;
            itaup  = itauq + m;
            nwork  = itaup + m;

            // Bidiagonalize A
            // Workspace:  need   2*M [tauq, taup] + N        [gebrd work]
            // Workspace:  prefer 2*M [tauq, taup] + (M+N)*NB [gebrd work]
            // RWorkspace: need   M [e]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_zgebrd( m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
            #endif

            if (want_qn) {                                        //
                // Path 5tn (N >> M, JOBZ='N')
                zgesdd_path = "5tn";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   M [e] + 4*M [bdsdc work]
                lapackf77_dbdsdc( "L", "N", &m, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 5to (N >> M, JOBZ='O')
                zgesdd_path = "5to";
                ivt = nwork;

                // Copy A to U, generate Q
                // Workspace:  need   2*M [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "L", &m, &m, A(1,1), &lda, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, m, n, U, ldu, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                // Generate P**H in A
                // Workspace:  need   2*M [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, m, n, m, A(1,1), lda, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                ldwrkvt = m;
                if (lwork >= m*n + wrkbl) {
                    // WORK[ IVT ] is M by N
                    nwork = ivt + ldwrkvt*n;
                    chunk = n;
                }
                else {
                    // WORK[ IVT ] is M by CHUNK
                    chunk = (lwork - wrkbl) / m;
                    assert( chunk >= m );
                    nwork = ivt + ldwrkvt*chunk;
                }

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply Q in U by real matrix RWORK[IRVT]
                // storing the result in WORK[IVT], copying to U
                // Workspace:  need   2*M [tauq, taup] + M*M [VT]
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [lacrm work]
                lapackf77_zlacrm( &m, &m, U, &ldu, &rwork[iru], &m, &work[ivt], &ldwrkvt, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &m, &work[ivt], &ldwrkvt, U, &ldu );

                // Multiply RWORK[IRVT] by P**H in A,
                // storing the result in WORK[IVT], copying to A
                // Workspace:  need   2*M [tauq, taup] + M*M [VT]
                // Workspace:  prefer 2*M [tauq, taup] + M*N [VT]
                // RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [larcm work]
                // RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [larcm work] < M + 5*M*M since N < 2*M here
                nrwork = iru;
                for (i = 1; i <= n; i += chunk) {
                    ib = min( n - i + 1, chunk );
                    lapackf77_zlarcm( &m, &ib, &rwork[irvt], &m, A(1,i), &lda, &work[ivt], &ldwrkvt, &rwork[nrwork] );
                    lapackf77_zlacpy( "F", &m, &ib, &work[ivt], &ldwrkvt, A(1,i), &lda );
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 5ts (N >> M, JOBZ='S')
                zgesdd_path = "5ts";
                // Copy A to U, generate Q
                // Workspace:  need   2*M [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "L", &m, &m, A(1,1), &lda, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, m, n, U, ldu, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                // Copy A to VT, generate P**H
                // Workspace:  need   2*M [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "U", &m, &n, A(1,1), &lda, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &m, &n, &m, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, m, n, m, VT, ldvt, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply Q in U by real matrix RWORK[IRU],
                // storing the result in A, copying to U
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [lacrm work]
                lapackf77_zlacrm( &m, &m, U, &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &m, A(1,1), &lda, U, &ldu );

                // Multiply real matrix RWORK[IRVT] by P**H in VT,
                // storing the result in A, copying to VT
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [larcm work] < M + 5*M*M since N < 2*M here
                nrwork = iru;
                lapackf77_zlarcm( &m, &n, &rwork[irvt], &m, VT, &ldvt, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, VT, &ldvt );
            }                                                     //
            else if (want_qa) {                                   //
                // Path 5ta (N >> M, JOBZ='A')
                zgesdd_path = "5ta";
                // Copy A to U, generate Q
                // Workspace:  need   2*M [tauq, taup] + M    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "L", &m, &m, A(1,1), &lda, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaQ, m, m, n, U, ldu, &work[itauq], &work[nwork], lnwork, &ierr );
                #endif

                // Copy A to VT, generate P**H
                // Workspace:  need   2*M [tauq, taup] + N    [ungbr work]
                // Workspace:  prefer 2*M [tauq, taup] + N*NB [ungbr work]
                // RWorkspace: need   0
                lapackf77_zlacpy( "U", &m, &n, A(1,1), &lda, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zungbr( "P", &n, &n, &m, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr );
                #else
                magma_zungbr( MagmaP, n, n, m, VT, ldvt, &work[itaup], &work[nwork], lnwork, &ierr );
                #endif

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Multiply Q in U by real matrix RWORK[IRU],
                // storing the result in A, copying to U
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [lacrm work]
                lapackf77_zlacrm( &m, &m, U, &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &m, A(1,1), &lda, U, &ldu );

                // Multiply real matrix RWORK[IRVT] by P**H in VT,
                // storing the result in A, copying to VT
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [larcm work] < M + 5*M*M since N < 2*M here
                // LAPACK doesn't reset nrwork here, so it needs an extra M*M:
                // ([M] + 2*M*M + 2*M*N) < [M] + 6*M*M since N < 2*M here */
                nrwork = iru;
                lapackf77_zlarcm( &m, &n, &rwork[irvt], &m, VT, &ldvt, A(1,1), &lda, &rwork[nrwork] );
                lapackf77_zlacpy( "F", &m, &n, A(1,1), &lda, VT, &ldvt );
            }                                                     //
        }                                                         //
        else {                                                    //
            // N < MNTHR2
            // Path 6t (N > M, but not much larger)
            zgesdd_path = "6t";
            // Reduce to bidiagonal form without LQ decomposition
            // Use ZUNMBR to compute singular vectors
            ie     = 1;
            nrwork = ie + m;
            itauq  = 1;
            itaup  = itauq + m;
            nwork  = itaup + m;

            // Bidiagonalize A
            // Workspace:  need   2*M [tauq, taup] + N        [gebrd work]
            // Workspace:  prefer 2*M [tauq, taup] + (M+N)*NB [gebrd work]
            // RWorkspace: need   M [e]
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_zgebrd( &m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr );
            #else
            magma_zgebrd( m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr );
            #endif
            
            if (want_qn) {                                        //
                // Path 6tn (N > M, JOBZ='N')
                zgesdd_path = "6tn";
                // Perform bidiagonal SVD, computing singular values only
                // Workspace:  need   0
                // RWorkspace: need   M [e] + 4*M [bdsdc work]
                lapackf77_dbdsdc( "L", "N", &m, s, &rwork[ie], rdummy, &ione, rdummy, &ione, rdummy, idummy, &rwork[nrwork], iwork, info );
            }                                                     //
            else if (want_qo) {                                   //
                // Path 6to (N > M, JOBZ='O')
                zgesdd_path = "6to";
                ldwrkvt = m;
                ivt     = nwork;
                if (lwork >= m*n + wrkbl) {
                    // WORK[ IVT ] is M by N
                    lapackf77_zlaset( "F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwrkvt );
                    nwork  = ivt + ldwrkvt*n;
                    chunk  = -1;
                }
                else {
                    // WORK[ IVT ] is M by CHUNK
                    chunk  = (lwork - wrkbl) / m;
                    assert( chunk >= m );  // needed? could be nb?
                    nwork  = ivt + ldwrkvt*chunk;
                }

                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of A
                // Workspace:  need   2*M [tauq, taup] + M*M [VT] + M    [unmbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*M [VT] + M*NB [unmbr work]
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                if (lwork >= m*n + wrkbl) {
                    // Path 6to-fast
                    zgesdd_path = "6to-fast";
                    // Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT]
                    // Overwrite WORK[IVT] by right singular vectors of A,
                    // copying to A
                    // Workspace:  need   2*M [tauq, taup] + M*N [VT] + M    [unmbr work]
                    // Workspace:  prefer 2*M [tauq, taup] + M*N [VT] + M*NB [unmbr work]
                    // RWorkspace: need   M [e] + M*M [RVT]
                    lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwrkvt );
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_zunmbr( "P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwrkvt, &work[nwork], &lnwork, &ierr );
                    #else
                    magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, m, n, m, A(1,1), lda, &work[itaup], &work[ivt], ldwrkvt, &work[nwork], lnwork, &ierr );
                    #endif
                    lapackf77_zlacpy( "F", &m, &n, &work[ivt], &ldwrkvt, A(1,1), &lda );
                }
                else {
                    // Path 6to-slow
                    zgesdd_path = "6to-slow";
                    // Generate P**H in A
                    // Workspace:  need   2*M [tauq, taup] + M*M [VT] + M    [ungbr work]
                    // Workspace:  prefer 2*M [tauq, taup] + M*M [VT] + M*NB [ungbr work]
                    // RWorkspace: need   0
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_zungbr( "P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr );
                    #else
                    magma_zungbr( MagmaP, m, n, m, A(1,1), lda, &work[itaup], &work[nwork], lnwork, &ierr );
                    #endif

                    // Multiply Q in A by real matrix RWORK[IRU],
                    // storing the result in WORK[IU], copying to A
                    // Workspace:  need   2*M [tauq, taup] + M*M [VT]
                    // Workspace:  prefer 2*M [tauq, taup] + M*N [VT]
                    // RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [larcm work]
                    // RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [larcm work] < M + 5*M*M since N < 2*M here
                    nrwork = iru;
                    for (i = 1; i <= n; i += chunk) {
                        ib = min( n - i + 1, chunk );
                        lapackf77_zlarcm( &m, &ib, &rwork[irvt], &m, A(1,i), &lda, &work[ivt], &ldwrkvt, &rwork[nrwork] );
                        lapackf77_zlacpy( "F", &m, &ib, &work[ivt], &ldwrkvt, A(1,i), &lda );
                    }
                }
            }                                                     //
            else if (want_qs) {                                   //
                // Path 6ts (N > M, JOBZ='S')
                zgesdd_path = "6ts";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of A
                // Workspace:  need   2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of A
                // Workspace:  need   2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   M [e] + M*M [RVT]
                lapackf77_zlaset( "F", &m, &n, &c_zero, &c_zero, VT, &ldvt );
                lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, m, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
            else if (want_qa) {                                   //
                // Path 6ta (N > M, JOBZ='A')
                zgesdd_path = "6ta";
                // Perform bidiagonal SVD,
                // computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and
                // computing right singular vectors of bidiagonal matrix in RWORK[IRVT]
                // Workspace:  need   0
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + (3*M*M + 4*M) [bdsdc work]
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_dbdsdc( "L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, rdummy, idummy, &rwork[nrwork], iwork, info );

                // Copy real matrix RWORK[IRU] to complex matrix U
                // Overwrite U by left singular vectors of A
                // Workspace:  need   2*M [tauq, taup] + M    [unmbr work]
                // Workspace:  prefer 2*M [tauq, taup] + M*NB [unmbr work]
                // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
                lapackf77_zlacp2( "F", &m, &m, &rwork[iru], &m, U, &ldu );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr );
                #endif

                // Set all of VT to identity matrix
                lapackf77_zlaset( "F", &n, &n, &c_zero, &c_one, VT, &ldvt );

                // Copy real matrix RWORK[IRVT] to complex matrix VT
                // Overwrite VT by right singular vectors of A
                // Workspace:  need   2*M [tauq, taup] + N    [unmbr work]
                // Workspace:  prefer 2*M [tauq, taup] + N*NB [unmbr work]
                // RWorkspace: need   M [e] + M*M [RVT]
                lapackf77_zlacp2( "F", &m, &m, &rwork[irvt], &m, VT, &ldvt );
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_zunmbr( "P", "R", "C", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr );
                #else
                magma_zunmbr( MagmaP, MagmaRight, MagmaConjTrans, n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr );
                #endif
            }                                                     //
        }                                                         //
    }                                                             //

    // Undo scaling if necessary
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl( "G", &izero, &izero, &bignum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
        if (*info != 0 && anrm > bignum) {
            magma_int_t minmn_1 = minmn - 1;
            lapackf77_dlascl( "G", &izero, &izero, &bignum, &anrm, &minmn_1, &ione, &rwork[ie], &minmn, &ierr );
        }
        if (anrm < smlnum) {
            lapackf77_dlascl( "G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
        if (*info != 0 && anrm < smlnum) {
            magma_int_t minmn_1 = minmn - 1;
            lapackf77_dlascl( "G", &izero, &izero, &smlnum, &anrm, &minmn_1, &ione, &rwork[ie], &minmn, &ierr );
        }
    }

    // Return optimal workspace in WORK[1] (Fortran index)
    work[1] = magma_zmake_lwork( maxwrk );

    return *info;
} // magma_zgesdd
