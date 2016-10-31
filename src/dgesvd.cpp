/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
    
       @author Stan Tomov
       @author Mark Gates
       @precisions normal d -> s

*/
#include "magma_internal.h"

#define REAL

const char* dgesvd_path = "none";

// Version 1 - LAPACK
// Version 2 - MAGMA
#define VERSION 2

/***************************************************************************//**
    Purpose
    -------
    DGESVD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and/or right singular
    vectors. The SVD is written
        
        A = U * SIGMA * transpose(V)
    
    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.
    
    Note that the routine returns VT = V**T, not V.
    
    Arguments
    ---------
    @param[in]
    jobu    magma_vec_t
            Specifies options for computing all or part of the matrix U:
      -     = MagmaAllVec:        all M columns of U are returned in array U:
      -     = MagmaSomeVec:       the first min(m,n) columns of U (the left singular
                                  vectors) are returned in the array U;
      -     = MagmaOverwriteVec:  the first min(m,n) columns of U (the left singular
                                  vectors) are overwritten on the array A;
      -     = MagmaNoVec:         no columns of U (no left singular vectors) are
                                  computed.
    
    @param[in]
    jobvt   magma_vec_t
            Specifies options for computing all or part of the matrix V**T:
      -     = MagmaAllVec:        all N rows of V**T are returned in the array VT;
      -     = MagmaSomeVec:       the first min(m,n) rows of V**T (the right singular
                                  vectors) are returned in the array VT;
      -     = MagmaOverwriteVec:  the first min(m,n) rows of V**T (the right singular
                                  vectors) are overwritten on the array A;
      -     = MagmaNoVec:         no rows of V**T (no right singular vectors) are
                                  computed.
    \n
            JOBVT and JOBU cannot both be MagmaOverwriteVec.
    
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
      -     if JOBU = MagmaOverwriteVec,  A is overwritten with the first min(m,n)
            columns of U (the left singular vectors, stored columnwise);
      -     if JOBVT = MagmaOverwriteVec, A is overwritten with the first min(m,n)
            rows of V**T (the right singular vectors, stored rowwise);
      -     if JOBU != MagmaOverwriteVec and JOBVT != MagmaOverwriteVec,
            the contents of A are destroyed.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).
    
    @param[out]
    U       DOUBLE PRECISION array, dimension (LDU,UCOL)
            (LDU,M) if JOBU = MagmaAllVec or (LDU,min(M,N)) if JOBU = MagmaSomeVec.
      -     If JOBU = MagmaAllVec, U contains the M-by-M orthogonal matrix U;
      -     if JOBU = MagmaSomeVec, U contains the first min(m,n) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBU = MagmaNoVec or MagmaOverwriteVec, U is not referenced.
    
    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBU = MagmaSomeVec or MagmaAllVec, LDU >= M.
    
    @param[out]
    VT      DOUBLE PRECISION array, dimension (LDVT,N)
      -     If JOBVT = MagmaAllVec, VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBVT = MagmaSomeVec, VT contains the first min(m,n) rows of V**T
            (the right singular vectors, stored rowwise);
      -     if JOBVT = MagmaNoVec or MagmaOverwriteVec, VT is not referenced.
    
    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1;
      -     if JOBVT = MagmaAllVec, LDVT >= N;
      -     if JOBVT = MagmaSomeVec, LDVT >= min(M,N).
    
    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the required LWORK.
            if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
            superdiagonal elements of an upper bidiagonal matrix B
            whose diagonal is in S (not necessarily sorted). B
            satisfies A = U * B * VT, so it has the same singular values
            as A, and singular vectors related by U and VT.
            
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= 1.6*mn.
            For job: N=None, O=Overwrite, S=Some, A=All.
            Paths below assume M >= N; for N > M swap jobu and jobvt.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_dgesvd_nb(M,N).
            For many cases, there is a fast algorithm, and a slow algorithm that
            uses less workspace. Here are sizes for both cases.
    \n
            Optimal lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any                  3*mn + 2*mn*nb
            Path 2:   jobu=O, jobvt=N        mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 3:   jobu=O, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 4:   jobu=S, jobvt=N        mn*mn +     3*mn + 2*mn*nb
            Path 5:   jobu=S, jobvt=O      2*mn*mn +     3*mn + 2*mn*nb
            Path 6:   jobu=S, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
            Path 7:   jobu=A, jobvt=N        mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  3*mn + (mx + mn)*nb
    \n
            Optimal lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2:   jobu=O, jobvt=N      3*mn + (mx + mn)*nb
            Path 3-9:                      3*mn + max(2*mn*nb, mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any              5*mn
            Path 2:   jobu=O, jobvt=N        mn*mn + 5*mn
            Path 3:   jobu=O, jobvt=A,S      mn*mn + 5*mn
            Path 4:   jobu=S, jobvt=N        mn*mn + 5*mn
            Path 5:   jobu=S, jobvt=O      2*mn*mn + 5*mn
            Path 6:   jobu=S, jobvt=A,S      mn*mn + 5*mn
            Path 7:   jobu=A, jobvt=N        mn*mn + max(5*mn, mn + mx)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(5*mn, mn + mx)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(5*mn, mn + mx)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  max(3*mn + mx, 5*mn)
    \n
            LAPACK minimum lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2-9:                      max(3*mn + mx, 5*mn)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if DBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of WORK
                above for details.
    
    @ingroup magma_gesvd
*******************************************************************************/
extern "C" magma_int_t
magma_dgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda, double *s,
    double *U,    magma_int_t ldu,
    double *VT,   magma_int_t ldvt,
    double *work, magma_int_t lwork,
    magma_int_t *info )
{
    dgesvd_path = "init";
    
    #define A(i_,j_)  (A  + (i_) + (j_)*lda)
    #define U(i_,j_)  (U  + (i_) + (j_)*ldu)
    #define VT(i_,j_) (VT + (i_) + (j_)*ldvt)
    
    // Constants
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;
    const magma_int_t izero    = 0;
    const magma_int_t ione     = 1;
    const magma_int_t ineg_one = -1;
    
    // System generated locals
    magma_int_t lwork2, m_1, n_1;
    
    // Local variables
    magma_int_t i, ie, ir, iu, ib, ncu;
    double dummy[1], unused[1], eps;
    magma_int_t nru, iscl;
    magma_int_t ierr, itau, ncvt, nrvt;
    magma_int_t chunk, minmn, wrkbl, itaup, itauq, mnthr, iwork;
    magma_int_t ldwrkr, ldwrku, maxwrk, minwrk, gemm_nb;
    double anrm, bignum, smlnum;
    
    // Parameter adjustments for Fortran indexing
    --work;
    
    // Function Body
    *info = 0;
    minmn = min( m, n );
    ie = 1;
    
    const bool want_ua  = (jobu == MagmaAllVec);
    const bool want_us  = (jobu == MagmaSomeVec);
    const bool want_uo  = (jobu == MagmaOverwriteVec);
    const bool want_un  = (jobu == MagmaNoVec);
    const bool want_uas = (want_ua || want_us);
    
    const bool want_va  = (jobvt == MagmaAllVec);
    const bool want_vs  = (jobvt == MagmaSomeVec);
    const bool want_vo  = (jobvt == MagmaOverwriteVec);
    const bool want_vn  = (jobvt == MagmaNoVec);
    const bool want_vas = (want_va || want_vs);
    
    const bool lquery = (lwork == -1);
    
    // Test the input arguments
    if (! (want_ua || want_us || want_uo || want_un)) {
        *info = -1;
    } else if (! (want_va || want_vs || want_vo || want_vn) || (want_vo && want_uo) ) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,m)) {
        *info = -6;
    } else if ((ldu < 1) || (want_uas && (ldu < m)) ) {
        *info = -9;
    } else if ((ldvt < 1) || (want_va && (ldvt < n)) || (want_vs && (ldvt < minmn)) ) {
        *info = -11;
    }
    
    gemm_nb = 64;
    
    // Compute workspace
    // (Note: Comments in the code beginning "Workspace:" describe the
    //  minimal amount of workspace needed at that point in the code,
    //  as well as the preferred amount for good performance.
    //  NB refers to the optimal block size for the immediately
    //  following subroutine, as returned by ILAENV or magma_get_*_nb.)
    minwrk = 1;
    maxwrk = 1;
    wrkbl  = 1;
    mnthr  = magma_int_t( minmn * 1.6 );
    if (*info == 0) {
        if (m >= n && minmn > 0) {
            // Compute space needed for DBDSQR
            // LAPACK adds n for [e] here; that is separated here for consistency
            magma_int_t lwork_dbdsqr = 4*n;
            
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
            lapackf77_dorgbr( "P", &n, &n, &n, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaP,   n,  n,  n, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_p_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "Q", &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaQ,   m,  n,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_q_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "Q", &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaQ,   m,  m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_q_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "Q", &n, &n, &n, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaQ,   n,  n,  n, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_q_nn = magma_int_t( real( dummy[0] ));
            
            // magma_dorgqr2 does not take workspace; use LAPACK's for compatability
            lapackf77_dorgqr( &m, &m, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_dorgqr_mm = magma_int_t( real( dummy[0] ));
            
            lapackf77_dorgqr( &m, &n, &n, unused, &m, unused, dummy, &ineg_one, &ierr );
            magma_int_t lwork_dorgqr_mn = magma_int_t( real( dummy[0] ));
            
            // missing from LAPACK, since it occurs only in slow paths
            #if VERSION == 1
            lapackf77_dormbr( "Q", "R", "N",                &m, &n, &n, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_qrn_mn = magma_int_t( real( dummy[0] ));
            
            // wrkbl is everything except R and U matrices.
            // It is used later to compute ldwrkr for R and ldwrku for U.
            // For minwrk, LAPACK used min workspace for each routine (usually m or n);
            // but that size doesn't work for MAGMA, as m*nb or n*nb is usually the min,
            // so here we track MAGMA's min workspace.
            if (m >= mnthr) {
                if (want_un) {
                    // Path 1 (M >> N, JOBU='N')
                    wrkbl  = max( wrkbl,      n + lwork_dgeqrf_mn   );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,    3*n + lwork_dgebrd_nn   );  // gebrd  = n or 2*n*nb
                    if (want_vo || want_vas) {
                        wrkbl = max( wrkbl, 3*n + lwork_dorgbr_p_nn );   // orgbr  = n or n*nb
                    }
                    wrkbl  = max( wrkbl,      n + lwork_dbdsqr      );  // bdsqr  = 4*n
                    maxwrk = wrkbl;                                     // maxwrk = 3*n + 2*n*nb
                    minwrk = maxwrk;                                    // minwrk = 3*n + 2*n*nb
                    //                                              lapack minwrk = 5*n
                }
                else if (want_uo && want_vn) {
                    // Path 2 (M >> N, JOBU='O', JOBVT='N')
                    // Path 2-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn      );  // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    // todo: is n*gemm_nb enough? LAPACK has m*n
                    maxwrk = n*n + max( wrkbl, n + n*gemm_nb        );  // maxwrk = n*n + 3*n + 2*n*nb
                    //                                              lapack maxwrk = n*n + max(3*n + 2*n*nb, n + m*n)
                    //                                              lapack minwrk = n*n + 5*n
                    
                    // Path 2-slow
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_mn     );  // gebrd  = m or (m+n)*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_q_mn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    // if (m+n)*nb (slow) > n*n + n*nb (fast),
                    // it's possible that fast subpath requires less workspace
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + (m+n)*nb
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_uo && want_vas) {
                    // Path 3 (M >> N, JOBU='O', JOBVT='S' or 'A')
                    // Path 3-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn      );  // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    // todo: is n*gemm_nb enough? LAPACK has m*n
                    maxwrk = n*n + max( wrkbl, n + n*gemm_nb  );        // maxwrk = n*n + 3*n + 2*n*nb
                    //                                              lapack maxwrk = n*n + max(3*n + 2*n*nb, n + m*n)
                    //                                              lapack minwrk = n*n + 5*n
                    
                    // Path 3-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mn     );  // orgqr  = n or n*nb
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_p_nn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_us && want_vn) {
                    // Path 4 (M >> N, JOBU='S', JOBVT='N')
                    // Path 4-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn      );  // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 3*n + 2*n*nb
                    //                                              lapack minwrk = n*n + 5*n
                                                                    
                    // Path 4-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mn     );  // orgqr  = n or n*nb
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_us && want_vo) {
                    // Path 5 (M >> N, JOBU='S', JOBVT='O')
                    // Path 5-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn      );  // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = 2*n*n + wrkbl;                             // maxwrk = 2*n*n + 3*n + 2*n*nb
                    //                                              lapack minwrk = 2*n*n + 5*n
                    
                    // Path 5-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mn     );  // orgqr  = n or n*nb
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_p_nn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_us && want_vas) {
                    // Path 6 (M >> N, JOBU='S', JOBVT='S' or 'A')
                    // Path 6-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mn      );  // orgqr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 3*n + 2*n*nb
                    //                                              lapack minwrk = n*n + 5*n
                    
                    // Path 6-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mn     );  // orgqr  = n or n*nb
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_p_nn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_ua && want_vn) {
                    // Path 7 (M >> N, JOBU='A', JOBVT='N')
                    // Path 7-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mm      );  // orgqr  = m or m*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(n + m*nb, 3*n + 2*n*nb)
                    //                                              lapack minwrk = n*n + max(m + n, 5*n)
                    
                    // Path 7-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mm     );  // orgqr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_ua && want_vo) {
                    // Path 8 (M >> N, JOBU='A', JOBVT='O')
                    // Path 8-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mm      );  // orgqr  = m or m*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = 2*n*n + wrkbl;                             // maxwrk = 2*n*n + max(n + m*nb, 3*n + 2*n*nb)
                    //                                              lapack minwrk = 2*n*n + max(m + n, 5*n)
                    
                    // Path 8-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mm     );  // orgqr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_p_nn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
                else if (want_ua && want_vas) {
                    // Path 9 (M >> N, JOBU='A', JOBVT='S' or 'A')
                    // Path 9-fast
                    wrkbl  = max( wrkbl,   n + lwork_dgeqrf_mn      );  // geqrf  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dorgqr_mm      );  // orgqr  = m or m*nb (note m)
                    wrkbl  = max( wrkbl, 3*n + lwork_dgebrd_nn      );  // gebrd  = n or 2*n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                    wrkbl  = max( wrkbl,   n + lwork_dbdsqr         );  // bdsqr  = 4*n
                    maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(n + m*nb, 3*n + 2*n*nb)
                    //                                              lapack minwrk = n*n + max(m + n, 5*n)
                    
                    // Path 9-slow
                    minwrk = max( minwrk,   n + lwork_dgeqrf_mn     );  // geqrf  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dorgqr_mm     );  // orgqr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dgebrd_nn     );  // gebrd  = n or 2*n*nb
                    minwrk = max( minwrk, 3*n + lwork_dormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
                    minwrk = max( minwrk, 3*n + lwork_dorgbr_p_nn   );  // orgbr  = n or n*nb
                    minwrk = max( minwrk,   n + lwork_dbdsqr        );  // bdsqr  = 4*n
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
                    //                                              lapack minwrk = max(3*n + m, 5*n)
                }
            }
            else {
                // Path 10 (M >= N, but not much larger)
                wrkbl  = max( wrkbl,     3*n + lwork_dgebrd_mn      );  // gebrd  = m or (m+n)*nb (note m)
                if (want_us || want_uo) {
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_mn    );  // orgbr  = n or n*nb
                }
                if (want_ua) {
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb (note m)
                }
                if (want_vas || want_vo) {
                    wrkbl  = max( wrkbl, 3*n + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb
                }
                wrkbl  = max( wrkbl,       n + lwork_dbdsqr         );  // bdsqr  = 4*n
                maxwrk = wrkbl;                                         // maxwrk = 3*n + (m+n)*nb
                minwrk = maxwrk;                                        // minwrk = 3*n + (m+n)*nb
                //                                                  lapack minwrk = max(3*n + m, 5*n)
            }
        }
        else if (minmn > 0) {
            // m < n
            // Compute space needed for DBDSQR
            // needs n for [e] and 4*n for [work]
            magma_int_t lwork_dbdsqr = 4*m;
            
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
            lapackf77_dorgbr( "P", &m, &m, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaP,   m,  m,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_p_mm = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "P", &m, &n, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaP,   m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_p_mn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "P", &n, &n, &m, unused, &n, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaP,   n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_p_nn = magma_int_t( real( dummy[0] ));
            
            #if VERSION == 1
            lapackf77_dorgbr( "Q", &m, &m, &m, unused, &m, unused, dummy, &ineg_one, &ierr );
            #else
            magma_dorgbr( MagmaQ,   m,  m,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dorgbr_q_mm = magma_int_t( real( dummy[0] ));
            
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
            
            // missing from LAPACK, since it occurs only in slow paths
            #if VERSION == 1
            lapackf77_dormbr( "P", "L", "T",             &m, &n, &m, unused, &m, unused, unused, &m, dummy, &ineg_one, &ierr );
            #else
            magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
            #endif
            magma_int_t lwork_dormbr_plt_mn = magma_int_t( real( dummy[0] ));
            
            if (n >= mnthr) {
                if (want_vn) {
                    // Path 1t (N >> M, JOBVT='N')
                    wrkbl  = max( wrkbl,      m + lwork_dgelqf_mn   );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,    3*m + lwork_dgebrd_mm   );  // gebrd  = m or 2*m*nb
                    if (want_uo || want_uas) {
                        wrkbl = max( wrkbl, 3*m + lwork_dorgbr_q_mm );  // orgbr  = m or m*nb
                    }
                    wrkbl  = max( wrkbl,      m + lwork_dbdsqr      );  // bdsqr  = 4*m
                    maxwrk = wrkbl;                                     // maxwrk = 3*m + 2*m*nb
                    minwrk = maxwrk;                                    // minwrk = 3*m + 2*m*nb
                    //                                              lapack minwrk = 5*m
                }
                else if (want_vo && want_un) {
                    // Path 2t (N >> M, JOBU='N', JOBVT='O')
                    // Path 2t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn      );  // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    // todo: is m*gemm_nb enough? LAPACK has m*n
                    maxwrk = m*m + max( wrkbl, m + m*gemm_nb        );  // maxwrk = m*m + 3*m + 2*m*nb
                    //                                              lapack maxwrk = m*m + max(3*m + 2*m*nb, m + m*n)
                    //                                              lapack minwrk = m*m + 5*m
                                                                        
                    // Path 2t-slow
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mn     );  // gebrd  = n or (m+n)*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_p_mn   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + (m+n)*nb
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_vo && want_uas) {
                    // Path 3t (N >> M, JOBU='S' or 'A', JOBVT='O')
                    // Path 3t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn      );  // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    // todo: is m*gemm_nb enough? LAPACK has m*n
                    maxwrk = m*m + max( wrkbl, m + m*gemm_nb        );  // maxwrk = m*m + 3*m + 2*m*nb
                    //                                              lapack maxwrk = m*m + max(3*m + 2*m*nb, m + m*n)
                    //                                              lapack minwrk = m*m + 5*m
                    
                    // Path 3t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_mn     );  // orglq  = m or m*nb
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_q_mm   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_vs && want_un) {
                    // Path 4t (N >> M, JOBU='N', JOBVT='S')
                    // Path 4t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn      );  // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 3*m + 2*m*nb
                    //                                              lapack minwrk = m*m + 5*m
                                                                        
                    // Path 4t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_mn     );  // orglq  = m or m*nb
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_vs && want_uo) {
                    // Path 5t (N >> M, JOBU='O', JOBVT='S')
                    // Path 5t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn      );  // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = 2*m*m + wrkbl;                             // maxwrk = 2*m*m + 3*m + 2*m*nb
                    //                                              lapack minwrk = 2*m*m + 5*m
                                                                        
                    // Path 5t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_mn     );  // orglq  = m or m*nb
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_q_mm   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_vs && want_uas) {
                    // Path 6t (N >> M, JOBU='S' or 'A', JOBVT='S')
                    // Path 6t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_mn      );  // orglq  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 3*m + 2*m*nb
                    //                                              lapack minwrk = m*m + 5*m
                                                                        
                    // Path 6t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_mn     );  // orglq  = m or m*nb
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_q_mm   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_va && want_un) {
                    // Path 7t (N >> M, JOBU='N', JOBVT='A')
                    // Path 7t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_nn      );  // orglq  = n or n*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(m + n*nb, 3*m + 2*m*nb)
                    //                                              lapack minwrk = m*m + max(m + n, 5*m)
                    
                    // Path 7t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_nn     );  // orglq  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_va && want_uo) {
                    // Path 8t (N >> M, JOBU='O', JOBVT='A')
                    // Path 8t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_nn      );  // orglq  = n or n*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = 2*m*m + wrkbl;                             // maxwrk = 2*m*m + max(m + n*nb, 3*m + 2*m*nb)
                    //                                              lapack minwrk = 2*m*m + max(m + n, 5*m)
                                                                        
                    // Path 8t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_nn     );  // orglq  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_q_mm   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
                else if (want_va && want_uas) {
                    // Path 9t (N >> M, JOBU='S' or 'A', JOBVT='A')
                    // Path 9t-fast
                    wrkbl  = max( wrkbl,   m + lwork_dgelqf_mn      );  // gelqf  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dorglq_nn      );  // orglq  = n or n*nb (note n)
                    wrkbl  = max( wrkbl, 3*m + lwork_dgebrd_mm      );  // gebrd  = m or 2*m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                    wrkbl  = max( wrkbl,   m + lwork_dbdsqr         );  // bdsqr  = 4*m
                    maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(m + n*nb, 3*m + 2*m*nb)
                    //                                              lapack minwrk = m*m + max(m + n, 5*m)
                                                                        
                    // Path 9t-slow
                    minwrk = max( minwrk,   m + lwork_dgelqf_mn     );  // gelqf  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dorglq_nn     );  // orglq  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dgebrd_mm     );  // gebrd  = m or 2*m*nb
                    minwrk = max( minwrk, 3*m + lwork_dormbr_plt_mn );  // ormbr  = n or n*nb (note n)
                    minwrk = max( minwrk, 3*m + lwork_dorgbr_q_mm   );  // orgbr  = m or m*nb
                    minwrk = max( minwrk,   m + lwork_dbdsqr        );  // bdsqr  = 4*m
                    minwrk = min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
                    //                                              lapack minwrk = max(3*m + n, 5*m)
                }
            }
            else {
                // Path 10t (N > M, but not much larger)
                wrkbl  = max( wrkbl,     3*m + lwork_dgebrd_mn      );  // gebrd  = n or (m+n)*nb (note n)
                if (want_vs || want_vo) {
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_mn    );  // orgbr  = m or m*nb
                }
                if (want_va) {
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_p_nn    );  // orgbr  = n or n*nb (note n)
                }
                if (want_uas || want_uo) {
                    wrkbl  = max( wrkbl, 3*m + lwork_dorgbr_q_mm    );  // orgbr  = m or m*nb
                }
                wrkbl  = max( wrkbl, m + lwork_dbdsqr               );  // bdsqr  = 4*m
                maxwrk = wrkbl;                                         // maxwrk = 3*m + (m+n)*nb
                minwrk = maxwrk;                                        // minwrk = 3*m + (m+n)*nb
                //                                                  lapack minwrk = max(3*m + n, 5*m)
            }
        }
        assert( minwrk <= maxwrk );
        
        work[1] = magma_dmake_lwork( maxwrk );
        
        if (lwork < minwrk && ! lquery) {
            *info = -13;
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
    eps = lapackf77_dlamch( "P" );
    smlnum = magma_dsqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;
    
    // Scale A if max element outside range [SMLNUM,BIGNUM]
    anrm = lapackf77_dlange( "M", &m, &n, A, &lda, dummy );
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_dlascl( "G", &izero, &izero, &anrm, &smlnum, &m, &n, A, &lda, &ierr );
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_dlascl( "G", &izero, &izero, &anrm, &bignum, &m, &n, A, &lda, &ierr );
    }
    
    m_1 = m - 1;
    n_1 = n - 1;
    
    if (m >= n) {                                                 //
        // A has at least as many rows as columns.
        // If A has sufficiently more rows than columns, first reduce using
        // the QR decomposition (if sufficient workspace available)
        if (m >= mnthr) {                                         //
            if (want_un) {                                        //
                // Path 1 (M >> N, JOBU='N')
                dgesvd_path = "1n,nosa";
                // No left singular vectors to be computed
                itau  = 1;
                iwork = itau + n;
                
                // Compute A=Q*R
                // Workspace: need   N [tau] + N    [geqrf work]
                // Workspace: prefer N [tau] + N*NB [geqrf work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                #else
                magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                #endif
                
                // Zero out below R
                lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(1,0), &lda );
                ie    = 1;
                itauq = ie    + n;
                itaup = itauq + n;
                iwork = itaup + n;
                
                // Bidiagonalize R in A
                // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &n, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
                
                ncvt = 0;
                if (want_vo || want_vas) {                        //
                    // If right singular vectors desired, generate P**T.
                    // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    ncvt = n;
                }                                                 //
                iwork = ie + n;
                
                // Perform bidiagonal QR iteration, computing right
                // singular vectors of A in A if desired
                // Workspace: need     N [e] + 4*N [bdsqr work]
                lapackf77_dbdsqr( "U", &n, &ncvt, &izero, &izero, s, &work[ie], A, &lda, dummy, &ione, dummy, &ione, &work[iwork], info );
                
                // If right singular vectors desired in VT, copy them there
                if (want_vas) {
                    lapackf77_dlacpy( "F", &n, &n, A,  &lda, VT, &ldvt );
                }
            }                                                     //
            else if (want_uo && want_vn) {                        //
                // Path 2 (M >> N, JOBU='O', JOBVT='N')
                dgesvd_path = "2o,n";
                // N left singular vectors to be overwritten on A and
                // no right singular vectors to be computed
                if (lwork >= n*n + wrkbl) {
                    // Path 2-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "2o,n-fast";
                    if (lwork >= max(wrkbl, lda*n + n) + lda*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= max(wrkbl, lda*n + n) + n*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is N by N
                        ldwrku = lda;
                        ldwrkr = n;
                    }
                    else {
                        // WORK(IU) is LDWRKU by N
                        // WORK(IR) is N by N
                        ldwrku = (lwork - n*n - n) / n;
                        ldwrkr = n;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr*n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to WORK(IR) and zero out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir+1], &ldwrkr );
                    
                    // Generate Q in A
                    // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left vectors bidiagonalizing R
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR)
                    // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &n, &izero, s, &work[ie], dummy, &ione, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    iu = ie + n;
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IR), storing result in WORK(IU) and copying to A
                    // Workspace: need   N*N [R] + N [e] + N    [U]
                    // Workspace: prefer N*N [R] + N [e] + NB*N [U]
                    // Workspace: max    N*N [R] + N [e] + M*N  [U]
                    // TODO: [e] not need; can change lwork above
                    for (i = 1; i <= m; i += ldwrku) {
                        ib = min( m - i + 1, ldwrku );
                        blasf77_dgemm( "N", "N", &ib, &n, &n,
                                       &c_one,  A(i-1,0), &lda,
                                                &work[ir], &ldwrkr,
                                       &c_zero, &work[iu], &ldwrku );
                        lapackf77_dlacpy( "F", &ib, &n, &work[iu], &ldwrku, A(i-1,0), &lda );
                    }
                }
                else {
                    // Path 2-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "2o,n-slow";
                    ie    = 1;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize A
                    // Workspace: need   3*N [e, tauq, taup] + M        [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left vectors bidiagonalizing A
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &n, &n, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  n,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &m, &izero, s, &work[ie], dummy, &ione, A, &lda, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_uo && want_vas) {                       //
                // Path 3 (M >> N, JOBU='O', JOBVT='S' or 'A')
                dgesvd_path = "3o,sa";
                // N left singular vectors to be overwritten on A and
                // N right singular vectors to be computed in VT
                if (lwork >= n*n + wrkbl) {
                    // Path 3-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "3o,sa-fast";
                    if (lwork >= max(wrkbl, lda*n + n) + lda*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= max(wrkbl, lda*n + n) + n*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is N by N
                        ldwrku = lda;
                        ldwrkr = n;
                    }
                    else {
                        // WORK(IU) is LDWRKU by N
                        // WORK(IR) is N by N
                        ldwrku = (lwork - n*n - n) / n;
                        ldwrkr = n;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to VT, zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A,  &lda, VT, &ldvt );
                    if (n > 1) {
                        lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, VT(1,0), &ldvt );
                    }
                    
                    // Generate Q in A
                    // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT, copying result to WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, VT, &ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &n, &n, VT, &ldvt, &work[ir], &ldwrkr );
                    
                    // Generate left vectors bidiagonalizing R in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing R in VT
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR) and computing right
                    // singular vectors of R in VT
                    // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &n, &izero, s, &work[ie], VT, &ldvt, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    iu = ie + n;
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IR), storing result in WORK(IU) and copying to A
                    // Workspace: need   N*N [R] + N [e] + N    [U]
                    // Workspace: prefer N*N [R] + N [e] + NB*N [U]
                    // Workspace: max    N*N [R] + N [e] + M*N  [U]
                    // TODO: [e] not need; can change lwork above
                    for (i = 1; i <= m; i += ldwrku) {
                        ib = min( m - i + 1, ldwrku );
                        blasf77_dgemm( "N", "N", &ib, &n, &n,
                                       &c_one,  A(i-1,0), &lda,
                                                &work[ir], &ldwrkr,
                                       &c_zero, &work[iu], &ldwrku );
                        lapackf77_dlacpy( "F", &ib, &n, &work[iu], &ldwrku, A(i-1,0), &lda );
                    }
                }
                else {
                    // Path 3-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "3o,sa-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to VT, zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A,  &lda, VT, &ldvt );
                    if (n > 1) {
                        lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, VT(1,0), &ldvt );
                    }
                    
                    // Generate Q in A
                    // Workspace: need   N [tau] + N    [orgqr work]
                    // Workspace: prefer N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, VT, &ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in A by left vectors bidiagonalizing R
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, VT, &ldvt, &work[itauq], A, &lda, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], A,  lda, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing R in VT
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A and computing right
                    // singular vectors of A in VT
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &m, &izero, s, &work[ie], VT, &ldvt, A, &lda, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_us && want_vn) {                        //
                // Path 4 (M >> N, JOBU='S', JOBVT='N')
                dgesvd_path = "4s,n";
                // N left singular vectors to be computed in U and
                // no right singular vectors to be computed
                if (lwork >= n*n + wrkbl) {
                    // Path 4-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "4s,n-fast";
                    if (lwork >= wrkbl + lda*n) {
                        // WORK(IR) is LDA by N
                        ldwrkr = lda;
                    }
                    else {
                        // WORK(IR) is N by N
                        ldwrkr = n;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to WORK(IR), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir+1], &ldwrkr );
                    
                    // Generate Q in A
                    // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left vectors bidiagonalizing R in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR)
                    // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &n, &izero, s, &work[ie], dummy, &ione, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IR), storing result in U
                    // Workspace: need   N*N [R]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  A, &lda,
                                            &work[ir], &ldwrkr,
                                   &c_zero, U, &ldu );
                }
                else {
                    // Path 4-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "4s,n-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + N    [orgqr work]
                    // Workspace: prefer N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Zero out below R in A
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(1,0), &lda );
                    
                    // Bidiagonalize R in A
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left vectors bidiagonalizing R
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, A, &lda, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &m, &izero, s, &work[ie], dummy, &ione, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_us && want_vo) {                        //
                // Path 5 (M >> N, JOBU='S', JOBVT='O')
                dgesvd_path = "5s,o";
                // N left singular vectors to be computed in U and
                // N right singular vectors to be overwritten on A
                if (lwork >= 2*n*n + wrkbl) {
                    // Path 5-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "5s,o-fast";
                    if (lwork >= wrkbl + 2*lda*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= wrkbl + (lda + n)*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is N by N
                        ldwrku = lda;
                        ldwrkr = n;
                    }
                    else {
                        // WORK(IU) is N by N
                        // WORK(IR) is N by N
                        ldwrku = n;
                        ldwrkr = n;
                    }
                    iu    = 1;
                    ir    = iu + ldwrku * n;
                    itau  = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   2*N*N [U,R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to WORK(IU), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[iu+1], &ldwrku );
                    
                    // Generate Q in A
                    // Workspace: need   2*N*N [U,R] + N [tau] + N    [orgqr work]
                    // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IU), copying result to
                    // WORK(IR)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &n, &n, &work[iu], &ldwrku, &work[ir], &ldwrkr );
                    
                    // Generate left bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[iu], &ldwrku, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IU) and computing
                    // right singular vectors of R in WORK(IR)
                    // Workspace: need   2*N*N [U,R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &n, &izero, s, &work[ie], &work[ir], &ldwrkr, &work[iu], &ldwrku, dummy, &ione, &work[iwork], info );
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IU), storing result in U
                    // Workspace: need   2*N*N [U,R]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  A, &lda,
                                            &work[iu], &ldwrku,
                                   &c_zero, U, &ldu );
                    
                    // Copy right singular vectors of R to A
                    // Workspace: need   2*N*N [U,R]
                    lapackf77_dlacpy( "F", &n, &n, &work[ir], &ldwrkr, A, &lda );
                }
                else {
                    // Path 5-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "5s,o-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + N    [orgqr work]
                    // Workspace: prefer N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Zero out below R in A
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(1,0), &lda );
                    
                    // Bidiagonalize R in A
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left vectors bidiagonalizing R
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, A, &lda, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing R in A
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in A
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &m, &izero, s, &work[ie], A, &lda, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_us && want_vas) {                       //
                // Path 6 (M >> N, JOBU='S', JOBVT='S' or 'A')
                dgesvd_path = "6s,sa";
                // N left singular vectors to be computed in U and
                // N right singular vectors to be computed in VT
                if (lwork >= n*n + wrkbl) {
                    // Path 6-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "6s,sa-fast";
                    if (lwork >= wrkbl + lda*n) {
                        // WORK(IU) is LDA by N
                        ldwrku = lda;
                    }
                    else {
                        // WORK(IU) is N by N
                        ldwrku = n;
                    }
                    iu    = 1;
                    itau  = iu + ldwrku * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // Workspace: need   N*N [U] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [U] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy R to WORK(IU), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[iu+1], &ldwrku );
                    
                    // Generate Q in A
                    // Workspace: need   N*N [U] + N [tau] + N    [orgqr work]
                    // Workspace: prefer N*N [U] + N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IU), copying result to VT
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &n, &n, &work[iu], &ldwrku, VT, &ldvt );
                    
                    // Generate left bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[iu], &ldwrku, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in VT
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IU) and computing
                    // right singular vectors of R in VT
                    // Workspace: need   N*N [U] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &n, &izero, s, &work[ie], VT, &ldvt, &work[iu], &ldwrku, dummy, &ione, &work[iwork], info );
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IU), storing result in U
                    // Workspace: need   N*N [U]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  A, &lda,
                                            &work[iu], &ldwrku,
                                   &c_zero, U, &ldu );
                }
                else {
                    // Path 6-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "6s,sa-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + N    [orgqr work]
                    // Workspace: prefer N [tau] + N*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &n, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    // Copy R to VT, zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A,  &lda, VT, &ldvt );
                    if (n > 1) {
                        lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, VT(1,0), &ldvt );
                    }
                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, VT, &ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left bidiagonalizing vectors in VT
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, VT, &ldvt, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in VT
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in VT
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &m, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_ua && want_vn) {                        //
                // Path 7 (M >> N, JOBU='A', JOBVT='N')
                dgesvd_path = "7a,n";
                // M left singular vectors to be computed in U and
                // no right singular vectors to be computed
                if (lwork >= n*n + wrkbl) {
                    // Path 7-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "7a,n-fast";
                    if (lwork >= wrkbl + lda*n) {
                        // WORK(IR) is LDA by N
                        ldwrkr = lda;
                    }
                    else {
                        // WORK(IR) is N by N
                        ldwrkr = n;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Copy R to WORK(IR), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[ir+1], &ldwrkr );
                    
                    // Generate Q in U
                    // Workspace: need   N*N [R] + N [tau] + M    [orgqr work]
                    // Workspace: prefer N*N [R] + N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR)
                    // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &n, &izero, s, &work[ie], dummy, &ione, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    
                    // Multiply Q in U by left singular vectors of R in
                    // WORK(IR), storing result in A
                    // Workspace: need   N*N [R]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  U, &ldu,
                                            &work[ir], &ldwrkr,
                                   &c_zero, A, &lda );
                    
                    // Copy left singular vectors of A from A to U
                    lapackf77_dlacpy( "F", &m, &n, A, &lda, U, &ldu );
                }
                else {
                    // Path 7-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "7a,n-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + M    [orgqr work]
                    // Workspace: prefer N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Zero out below R in A
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(1,0), &lda );
                    
                    // Bidiagonalize R in A
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left bidiagonalizing vectors in A
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, A, &lda, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &izero, &m, &izero, s, &work[ie], dummy, &ione, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_ua && want_vo) {                        //
                // Path 8 (M >> N, JOBU='A', JOBVT='O')
                dgesvd_path = "8a,o";
                // M left singular vectors to be computed in U and
                // N right singular vectors to be overwritten on A
                if (lwork >= 2*n*n + wrkbl) {
                    // Path 8-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "8a,o-fast";
                    if (lwork >= wrkbl + 2*lda*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= wrkbl + (lda + n)*n) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is N by N
                        ldwrku = lda;
                        ldwrkr = n;
                    }
                    else {
                        // WORK(IU) is N by N
                        // WORK(IR) is N by N
                        ldwrku = n;
                        ldwrkr = n;
                    }
                    iu    = 1;
                    ir    = iu + ldwrku * n;
                    itau  = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   2*N*N [U,R] + N [tau] + N    [geqrf work]
                    // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   2*N*N [U,R] + N [tau] + M    [orgqr work]
                    // Workspace: prefer 2*N*N [U,R] + N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif
                    
                    // Copy R to WORK(IU), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[iu+1], &ldwrku );
                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IU), copying result to
                    // WORK(IR)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &n, &n, &work[iu], &ldwrku, &work[ir], &ldwrkr );
                    
                    // Generate left bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[iu], &ldwrku, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IU) and computing
                    // right singular vectors of R in WORK(IR)
                    // Workspace: need   2*N*N [U,R] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &n, &izero, s, &work[ie], &work[ir], &ldwrkr, &work[iu], &ldwrku, dummy, &ione, &work[iwork], info );
                    
                    // Multiply Q in U by left singular vectors of R in
                    // WORK(IU), storing result in A
                    // Workspace: need   N*N [U]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  U, &ldu,
                                            &work[iu], &ldwrku,
                                   &c_zero, A, &lda );
                    
                    // Copy left singular vectors of A from A to U
                    lapackf77_dlacpy( "F", &m, &n, A, &lda, U, &ldu );
                    
                    // Copy right singular vectors of R from WORK(IR) to A
                    lapackf77_dlacpy( "F", &n, &n, &work[ir], &ldwrkr, A, &lda );
                }
                else {
                    // Path 8-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "8a,o-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + M    [orgqr work]
                    // Workspace: prefer N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif

                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Zero out below R in A
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, A(1,0), &lda );
                    
                    // Bidiagonalize R in A
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left bidiagonalizing vectors in A
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, A, &lda, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in A
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in A
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &m, &izero, s, &work[ie], A, &lda, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_ua && want_vas) {                       //
                // Path 9 (M >> N, JOBU='A', JOBVT='S' or 'A')
                dgesvd_path = "9a,sa";
                // M left singular vectors to be computed in U and
                // N right singular vectors to be computed in VT
                if (lwork >= n*n + wrkbl) {
                    // Path 9-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "9a,sa-fast";
                    if (lwork >= wrkbl + lda*n) {
                        // WORK(IU) is LDA by N
                        ldwrku = lda;
                    }
                    else {
                        // WORK(IU) is N by N
                        ldwrku = n;
                    }
                    iu    = 1;
                    itau  = iu + ldwrku * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N*N [U] + N [tau] + N    [geqrf work]
                    // Workspace: prefer N*N [U] + N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N*N [U] + N [tau] + M    [orgqr work]
                    // Workspace: prefer N*N [U] + N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif
                    
                    // Copy R to WORK(IU), zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, &work[iu+1], &ldwrku );
                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IU), copying result to VT
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &n, &n, &work[iu], &ldwrku, VT, &ldvt );
                    
                    // Generate left bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &n, &n, &n, &work[iu], &ldwrku, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif

                    // Generate right bidiagonalizing vectors in VT
                    // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IU) and computing
                    // right singular vectors of R in VT
                    // Workspace: need   N*N [U] + N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &n, &izero, s, &work[ie], VT, &ldvt, &work[iu], &ldwrku, dummy, &ione, &work[iwork], info );

                    // Multiply Q in U by left singular vectors of R in
                    // WORK(IU), storing result in A
                    // Workspace: need   N*N [U]
                    blasf77_dgemm( "N", "N", &m, &n, &n,
                                   &c_one,  U, &ldu,
                                            &work[iu], &ldwrku,
                                   &c_zero, A, &lda );

                    // Copy left singular vectors of A from A to U
                    lapackf77_dlacpy( "F", &m, &n, A, &lda, U, &ldu );
                }
                else {
                    // Path 9-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "9a,sa-slow";
                    itau  = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R, copying result to U
                    // Workspace: need   N [tau] + N    [geqrf work]
                    // Workspace: prefer N [tau] + N*NB [geqrf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgeqrf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                    
                    // Generate Q in U
                    // Workspace: need   N [tau] + M    [orgqr work]
                    // Workspace: prefer N [tau] + M*NB [orgqr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr( &m, &m, &n, U, &ldu, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );
                    #endif
                    
                    // Copy R from A to VT, zeroing out below it
                    lapackf77_dlacpy( "U", &n, &n, A,  &lda, VT, &ldvt );
                    if (n > 1) {
                        lapackf77_dlaset( "L", &n_1, &n_1, &c_zero, &c_zero, VT(1,0), &ldvt );
                    }
                    ie    = itau;
                    itauq = ie    + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT
                    // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
                    // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &n, &n, VT, &ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply Q in U by left bidiagonalizing vectors in VT
                    // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "Q", "R", "N", &m, &n, &n, VT, &ldvt, &work[itauq], U, &ldu, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in VT
                    // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
                    // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in VT
                    // Workspace: need     N [e] + 4*N [bdsqr work]
                    lapackf77_dbdsqr( "U", &n, &n, &m, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
        }                                                         //
        else {                                                    //
            // M < MNTHR
            // Path 10 (M >= N, but not much larger)
            dgesvd_path = "10";
            // Reduce to bidiagonal form without QR decomposition
            ie    = 1;
            itauq = ie    + n;
            itaup = itauq + n;
            iwork = itaup + n;
            
            // Bidiagonalize A
            // Workspace: need   3*N [e, tauq, taup] + M        [gebrd work]
            // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [gebrd work]
            lwork2 = lwork - iwork + 1;

            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
            #else
            magma_dgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
            #endif
            
            if (want_uas) {                                       //
                // If left singular vectors desired in U, copy result to U
                // and generate left bidiagonalizing vectors in U
                // Workspace: need     3*N [e, tauq, taup] + NCU    [orgbr work]
                // Workspace: prefer   3*N [e, tauq, taup] + NCU*NB [orgbr work]
                lapackf77_dlacpy( "L", &m, &n, A, &lda, U, &ldu );
                if (want_us) {
                    ncu = n;
                }
                else {
                    assert(want_ua);
                    ncu = m;
                }
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "Q", &m, &ncu, &n, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaQ,   m,  ncu,  n, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_vas) {                                       //
                // If right singular vectors desired in VT, copy result to
                // VT and generate right bidiagonalizing vectors in VT
                // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
                // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
                lapackf77_dlacpy( "U", &n, &n, A,  &lda, VT, &ldvt );
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_uo) {                                        //
                // If left singular vectors desired in A, generate left
                // bidiagonalizing vectors in A
                // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
                // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "Q", &m, &n, &n, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaQ,   m,  n,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_vo) {                                        //
                // If right singular vectors desired in A, generate right
                // bidiagonalizing vectors in A
                // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
                // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "P", &n, &n, &n, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //

            iwork = ie + n;
            if (want_uas || want_uo) {
                nru = m;
            }
            if (want_un) {
                nru = 0;
            }
            if (want_vas || want_vo) {
                ncvt = n;
            }
            if (want_vn) {
                ncvt = 0;
            }
            if (! want_uo && ! want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in VT
                // Workspace: need       N [e] + 4*N [bdsqr work]
                lapackf77_dbdsqr( "U", &n, &ncvt, &nru, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
            }
            else if (! want_uo && want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in A
                // Workspace: need       N [e] + 4*N [bdsqr work]
                lapackf77_dbdsqr( "U", &n, &ncvt, &nru, &izero, s, &work[ie], A, &lda, U, &ldu, dummy, &ione, &work[iwork], info );
            }
            else {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in A and computing right singular
                // vectors in VT
                // Workspace: need       N [e] + 4*N [bdsqr work]
                lapackf77_dbdsqr( "U", &n, &ncvt, &nru, &izero, s, &work[ie], VT, &ldvt, A, &lda, dummy, &ione, &work[iwork], info );
            }
        }                                                         //
    }                                                             //
    else {                                                        //
        // m < n
        // A has more columns than rows.
        // If A has sufficiently more columns than rows, first reduce using
        // the LQ decomposition (if sufficient workspace available)
        if (n >= mnthr) {                                         //
            if (want_vn) {                                        //
                // Path 1t (N >> M, JOBVT='N')
                dgesvd_path = "1tnosa,n";
                // No right singular vectors to be computed
                itau  = 1;
                iwork = itau + m;
                
                // Compute A=L*Q
                // Workspace: need   M [tau] + M    [gelqf work]
                // Workspace: prefer M [tau] + M*NB [gelqf work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                #else
                magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                #endif
                
                // Zero out above L
                lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(0,1), &lda );
                ie    = 1;
                itauq = ie + m;
                itaup = itauq + m;
                iwork = itaup + m;
                
                // Bidiagonalize L in A
                // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd( &m, &m, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
                
                if (want_uo || want_uas) {                        //
                    // If left singular vectors desired, generate Q
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                }                                                 //
                iwork = ie + m;
                nru = 0;
                if (want_uo || want_uas) {
                    nru = m;
                }
                
                // Perform bidiagonal QR iteration, computing left singular
                // vectors of A in A if desired
                // Workspace: need     M [e] + 4*M [bdsqr work]
                lapackf77_dbdsqr( "U", &m, &izero, &nru, &izero, s, &work[ie], dummy, &ione, A, &lda, dummy, &ione, &work[iwork], info );
                
                // If left singular vectors desired in U, copy them there
                if (want_uas) {
                    lapackf77_dlacpy( "F", &m, &m, A, &lda, U, &ldu );
                }
            }                                                     //
            else if (want_vo && want_un) {                        //
                // Path 2t (N >> M, JOBU='N', JOBVT='O')
                dgesvd_path = "2tn,o";
                // M right singular vectors to be overwritten on A and
                // no left singular vectors to be computed
                if (lwork >= m*m + wrkbl) {
                    // Path 2t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "2tn,o-fast";
                    if (lwork >= max(wrkbl, lda*n + m) + lda*m) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by M
                        ldwrku = lda;
                        chunk  = n;
                        ldwrkr = lda;
                    }
                    else  if (lwork >= max(wrkbl, lda*n + m) + m*m) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is M by M
                        ldwrku = lda;
                        chunk  = n;
                        ldwrkr = m;
                    }
                    else {
                        // WORK(IU) is M by CHUNK
                        // WORK(IR) is M by M
                        ldwrku = m;
                        chunk  = (lwork - m*m - m) / m;
                        ldwrkr = m;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IR) and zero out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[ir+ldwrkr], &ldwrkr );
                    
                    // Generate Q in A
                    // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing L
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of L in WORK(IR)
                    // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &izero, &izero, s, &work[ie], &work[ir], &ldwrkr, dummy, &ione, dummy, &ione, &work[iwork], info );
                    iu = ie + m;
                    
                    // Multiply right singular vectors of L in WORK(IR) by Q
                    // in A, storing result in WORK(IU) and copying to A
                    // Workspace: need   M*M [R] + M [e] + M    [U]
                    // Workspace: prefer M*M [R] + M [e] + M*NB [U]
                    // Workspace: max    M*M [R] + M [e] + M*N  [U]
                    // TODO: [e] not need; can change lwork above
                    for (i = 1; i <= n; i += chunk) {
                        ib = min( n - i + 1, chunk );
                        blasf77_dgemm( "N", "N", &m, &ib, &m,
                                       &c_one,  &work[ir], &ldwrkr,
                                                A(0,i-1), &lda,
                                       &c_zero, &work[iu], &ldwrku );
                        lapackf77_dlacpy( "F", &m, &ib, &work[iu], &ldwrku, A(0,i-1), &lda );
                    }
                }
                else {
                    // Path 2t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "2tn,o-slow";
                    ie    = 1;
                    itauq = ie    + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize A
                    // Workspace: need   3*M [e, tauq, taup] + M        [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing A
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &n, &m, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  n,  m, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of A in A
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "L", &m, &n, &izero, &izero, s, &work[ie], A, &lda, dummy, &ione, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_vo && want_uas) {                       //
                // Path 3t (N >> M, JOBU='S' or 'A', JOBVT='O')
                dgesvd_path = "3tsa,o";
                // M right singular vectors to be overwritten on A and
                // M left singular vectors to be computed in U
                if (lwork >= m*m + wrkbl) {
                    // Path 3t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "3tsa,o-fast";
                    if (lwork >= max(wrkbl, lda*n + m) + lda*m) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is LDA by M
                        ldwrku = lda;
                        chunk  = n;
                        ldwrkr = lda;
                    }
                    else if (lwork >= max(wrkbl, lda*n + m) + m*m) {
                        // WORK(IU) is LDA by N
                        // WORK(IR) is M by M
                        ldwrku = lda;
                        chunk  = n;
                        ldwrkr = m;
                    }
                    else {
                        // WORK(IU) is M by CHUNK
                        // WORK(IR) is M by M
                        ldwrku = m;
                        chunk  = (lwork - m*m - m) / m;
                        ldwrkr = m;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to U, zeroing about above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, U, &ldu );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, U(0,1), &ldu );
                    
                    // Generate Q in A
                    // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U, copying result to WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, U, &ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &m, U, &ldu, &work[ir], &ldwrkr );
                    
                    // Generate right vectors bidiagonalizing L in WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left vectors bidiagonalizing L in U
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in U, and computing right
                    // singular vectors of L in WORK(IR)
                    // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &m, &izero, s, &work[ie], &work[ir], &ldwrkr, U, &ldu, dummy, &ione, &work[iwork], info );
                    iu = ie + m;
                    
                    // Multiply right singular vectors of L in WORK(IR) by Q
                    // in A, storing result in WORK(IU) and copying to A
                    // Workspace: need   M*M [R] + M [e] + M    [U]
                    // Workspace: prefer M*M [R] + M [e] + M*NB [U]
                    // Workspace: max    M*M [R] + M [e] + M*N  [U]
                    // TODO: [e] not need; can change lwork above
                    for (i = 1; i <= n; i += chunk) {
                        ib = min( n - i + 1, chunk );
                        blasf77_dgemm( "N", "N", &m, &ib, &m,
                                       &c_one,  &work[ir], &ldwrkr,
                                                A(0,i-1), &lda,
                                       &c_zero, &work[iu], &ldwrku );
                        lapackf77_dlacpy( "F", &m, &ib, &work[iu], &ldwrku, A(0,i-1), &lda );
                    }
                }
                else {
                    // Path 3t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "3tsa,o-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to U, zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, U, &ldu );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, U(0,1), &ldu );
                    
                    // Generate Q in A
                    // Workspace: need   M [tau] + M    [orglq work]
                    // Workspace: prefer M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, U, &ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right vectors bidiagonalizing L by Q in A
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, U, &ldu, &work[itaup], A, &lda, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], A,  lda, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left vectors bidiagonalizing L in U
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in A
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &m, &izero, s, &work[ie], A, &lda, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_vs && want_un) {                        //
                // Path 4t (N >> M, JOBU='N', JOBVT='S')
                dgesvd_path = "4tn,s";
                // M right singular vectors to be computed in VT and
                // no left singular vectors to be computed
                if (lwork >= m*m + wrkbl) {
                    // Path 4t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "4tn,s-fast";
                    if (lwork >= wrkbl + lda*m) {
                        // WORK(IR) is LDA by M
                        ldwrkr = lda;
                    }
                    else {
                        // WORK(IR) is M by M
                        ldwrkr = m;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IR), zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[ir+ldwrkr], &ldwrkr );
                    
                    // Generate Q in A
                    // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right vectors bidiagonalizing L in
                    // WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of L in WORK(IR)
                    // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &izero, &izero, s, &work[ie], &work[ir], &ldwrkr, dummy, &ione, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IR) by
                    // Q in A, storing result in VT
                    // Workspace: need   M*M [R]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[ir], &ldwrkr,
                                            A,  &lda,
                                   &c_zero, VT, &ldvt );
                }
                else {
                    // Path 4t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "4tn,s-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy result to VT
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + M    [orglq work]
                    // Workspace: prefer M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Zero out above L in A
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(0,1), &lda );
                    
                    // Bidiagonalize L in A
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right vectors bidiagonalizing L by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, A,  &lda, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &izero, &izero, s, &work[ie], VT, &ldvt, dummy, &ione, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_vs && want_uo) {                        //
                // Path 5t (N >> M, JOBU='O', JOBVT='S')
                dgesvd_path = "5to,s";
                // M right singular vectors to be computed in VT and
                // M left singular vectors to be overwritten on A
                if (lwork >= 2*m*m + wrkbl) {
                    // Path 5t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "5to,s-fast";
                    if (lwork >= wrkbl + 2*lda*m) {
                        // WORK(IU) is LDA by M
                        // WORK(IR) is LDA by M
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= wrkbl + (lda + m) * m) {
                        // WORK(IU) is LDA by M
                        // WORK(IR) is M by M
                        ldwrku = lda;
                        ldwrkr = m;
                    }
                    else {
                        // WORK(IU) is M by M
                        // WORK(IR) is M by M
                        ldwrku = m;
                        ldwrkr = m;
                    }
                    iu    = 1;
                    ir    = iu + ldwrku * m;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   2*M*M [U,R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IU), zeroing out below it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[iu+ldwrku], &ldwrku );
                    
                    // Generate Q in A
                    // Workspace: need   2*M*M [U,R] + M [tau] + M    [orglq work]
                    // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IU), copying result to
                    // WORK(IR)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &m, &work[iu], &ldwrku, &work[ir], &ldwrkr );
                    
                    // Generate right bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[iu], &ldwrku, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in WORK(IR) and computing
                    // right singular vectors of L in WORK(IU)
                    // Workspace: need   2*M*M [U,R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &m, &izero, s, &work[ie], &work[iu], &ldwrku, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IU) by
                    // Q in A, storing result in VT
                    // Workspace: need   2*M*M [U,R]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[iu], &ldwrku,
                                            A,  &lda,
                                   &c_zero, VT, &ldvt );
                    
                    // Copy left singular vectors of L to A
                    // Workspace: need   2*M*M [U,R]
                    lapackf77_dlacpy( "F", &m, &m, &work[ir], &ldwrkr, A, &lda );
                }
                else {
                    // Path 5t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "5to,s-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + M    [orglq work]
                    // Workspace: prefer M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Zero out above L in A
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(0,1), &lda );
                    
                    // Bidiagonalize L in A
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right vectors bidiagonalizing L by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, A,  &lda, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors of L in A
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A and computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &m, &izero, s, &work[ie], VT, &ldvt, A, &lda, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_vs && want_uas) {                       //
                // Path 6t (N >> M, JOBU='S' or 'A', JOBVT='S')
                dgesvd_path = "6tsa,s";
                // M right singular vectors to be computed in VT and
                // M left singular vectors to be computed in U
                if (lwork >= m*m + wrkbl) {
                    // Path 6t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "6tsa,s-fast";
                    if (lwork >= wrkbl + lda*m) {
                        // WORK(IU) is LDA by N
                        ldwrku = lda;
                    }
                    else {
                        // WORK(IU) is LDA by M
                        ldwrku = m;
                    }
                    iu    = 1;
                    itau  = iu + ldwrku * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // Workspace: need   M*M [U] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [U] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IU), zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[iu+ldwrku], &ldwrku );
                    
                    // Generate Q in A
                    // Workspace: need   M*M [U] + M [tau] + M    [orglq work]
                    // Workspace: prefer M*M [U] + M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IU), copying result to U
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &m, &work[iu], &ldwrku, U, &ldu );
                    
                    // Generate right bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[iu], &ldwrku, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in U
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in U and computing right
                    // singular vectors of L in WORK(IU)
                    // Workspace: need   M*M [U] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &m, &izero, s, &work[ie], &work[iu], &ldwrku, U, &ldu, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IU) by
                    // Q in A, storing result in VT
                    // Workspace: need   M*M [U]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[iu], &ldwrku,
                                            A,  &lda,
                                   &c_zero, VT, &ldvt );
                }
                else {
                    // Path 6t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "6tsa,s-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + M    [orglq work]
                    // Workspace: prefer M [tau] + M*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to U, zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, U, &ldu );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, U(0,1), &ldu );
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, U, &ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right bidiagonalizing vectors in U by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, U, &ldu, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in U
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &m, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_va && want_un) {                        //
                // Path 7t (N >> M, JOBU='N', JOBVT='A')
                dgesvd_path = "7tn,a";
                // N right singular vectors to be computed in VT and
                // no left singular vectors to be computed
                if (lwork >= m*m + wrkbl) {
                    // Path 7t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "7tn,a-fast";
                    if (lwork >= wrkbl + lda*m) {
                        // WORK(IR) is LDA by M
                        ldwrkr = lda;
                    }
                    else {
                        // WORK(IR) is M by M
                        ldwrkr = m;
                    }
                    ir    = 1;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Copy L to WORK(IR), zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[ir], &ldwrkr );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[ir+ldwrkr], &ldwrkr );
                    
                    // Generate Q in VT
                    // Workspace: need   M*M [R] + M [tau] + N    [orglq work]
                    // Workspace: prefer M*M [R] + M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate right bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[ir], &ldwrkr, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of L in WORK(IR)
                    // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &izero, &izero, s, &work[ie], &work[ir], &ldwrkr, dummy, &ione, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IR) by
                    // Q in VT, storing result in A
                    // Workspace: need   M*M [R]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[ir], &ldwrkr,
                                            VT, &ldvt,
                                   &c_zero, A, &lda );
                    
                    // Copy right singular vectors of A from A to VT
                    lapackf77_dlacpy( "F", &m, &n, A,  &lda, VT, &ldvt );
                }
                else {
                    // Path 7t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "7tn,a-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + N    [orglq work]
                    // Workspace: prefer M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Zero out above L in A
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(0,1), &lda );
                    
                    // Bidiagonalize L in A
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right bidiagonalizing vectors in A by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, A,  &lda, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &izero, &izero, s, &work[ie], VT, &ldvt, dummy, &ione, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_va && want_uo) {                        //
                // Path 8t (N >> M, JOBU='O', JOBVT='A')
                dgesvd_path = "8to,a";
                // N right singular vectors to be computed in VT and
                // M left singular vectors to be overwritten on A
                if (lwork >= 2*m*m + wrkbl) {
                    // Path 8t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "8to,a-fast";
                    if (lwork >= wrkbl + 2*lda*m) {
                        // WORK(IU) is LDA by M
                        // WORK(IR) is LDA by M
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else if (lwork >= wrkbl + (lda + m) * m) {
                        // WORK(IU) is LDA by M
                        // WORK(IR) is M by M
                        ldwrku = lda;
                        ldwrkr = m;
                    }
                    else {
                        // WORK(IU) is M by M
                        // WORK(IR) is M by M
                        ldwrku = m;
                        ldwrkr = m;
                    }
                    iu    = 1;
                    ir    = iu + ldwrku * m;
                    itau  = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   2*M*M [U,R] + M [tau] + M    [gelqf work]
                    // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   2*M*M [U,R] + M [tau] + N    [orglq work]
                    // Workspace: prefer 2*M*M [U,R] + M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IU), zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[iu+ldwrku], &ldwrku );
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IU), copying result to WORK(IR)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &m, &work[iu], &ldwrku, &work[ir], &ldwrkr );
                    
                    // Generate right bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[iu], &ldwrku, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in WORK(IR)
                    // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, &work[ir], &ldwrkr, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in WORK(IR) and computing
                    // right singular vectors of L in WORK(IU)
                    // Workspace: need   2*M*M [U,R] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &m, &izero, s, &work[ie], &work[iu], &ldwrku, &work[ir], &ldwrkr, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IU) by
                    // Q in VT, storing result in A
                    // Workspace: need   M*M [U]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[iu], &ldwrku,
                                            VT, &ldvt,
                                   &c_zero, A, &lda );
                    
                    // Copy right singular vectors of A from A to VT
                    lapackf77_dlacpy( "F", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Copy left singular vectors of A from WORK(IR) to A
                    lapackf77_dlacpy( "F", &m, &m, &work[ir], &ldwrkr, A, &lda );
                }
                else {
                    // Path 8t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "8to,a-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + N    [orglq work]
                    // Workspace: prefer M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Zero out above L in A
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, A(0,1), &lda );
                    
                    // Bidiagonalize L in A
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right bidiagonalizing vectors in A by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, A,  &lda, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in A
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A and computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &m, &izero, s, &work[ie], VT, &ldvt, A, &lda, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
            else if (want_va && want_uas) {                       //
                // Path 9t (N >> M, JOBU='S' or 'A', JOBVT='A')
                dgesvd_path = "9tsa,a";
                // N right singular vectors to be computed in VT and
                // M left singular vectors to be computed in U
                if (lwork >= m*m + wrkbl) {
                    // Path 9t-fast: Sufficient workspace for a fast algorithm
                    dgesvd_path = "9tsa,a-fast";
                    if (lwork >= wrkbl + lda*m) {
                        // WORK(IU) is LDA by M
                        ldwrku = lda;
                    }
                    else {
                        // WORK(IU) is M by M
                        ldwrku = m;
                    }
                    iu    = 1;
                    itau  = iu + ldwrku * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M*M [U] + M [tau] + M    [gelqf work]
                    // Workspace: prefer M*M [U] + M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M*M [U] + M [tau] + N    [orglq work]
                    // Workspace: prefer M*M [U] + M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to WORK(IU), zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, &work[iu], &ldwrku );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, &work[iu+ldwrku], &ldwrku );
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IU), copying result to U
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, &work[iu], &ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "L", &m, &m, &work[iu], &ldwrku, U, &ldu );
                    
                    // Generate right bidiagonalizing vectors in WORK(IU)
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "P", &m, &m, &m, &work[iu], &ldwrku, &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in U
                    // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in U and computing right
                    // singular vectors of L in WORK(IU)
                    // Workspace: need   M*M [U] + M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &m, &m, &izero, s, &work[ie], &work[iu], &ldwrku, U, &ldu, dummy, &ione, &work[iwork], info );
                    
                    // Multiply right singular vectors of L in WORK(IU) by
                    // Q in VT, storing result in A
                    // Workspace: need   M*M [U]
                    blasf77_dgemm( "N", "N", &m, &n, &m,
                                   &c_one,  &work[iu], &ldwrku,
                                            VT, &ldvt,
                                   &c_zero, A, &lda );
                    
                    // Copy right singular vectors of A from A to VT
                    lapackf77_dlacpy( "F", &m, &n, A,  &lda, VT, &ldvt );
                }
                else {
                    // Path 9t-slow: Insufficient workspace for a fast algorithm
                    dgesvd_path = "9tsa,a-slow";
                    itau  = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q, copying result to VT
                    // Workspace: need   M [tau] + M    [gelqf work]
                    // Workspace: prefer M [tau] + M*NB [gelqf work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgelqf( &m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                    
                    // Generate Q in VT
                    // Workspace: need   M [tau] + N    [orglq work]
                    // Workspace: prefer M [tau] + N*NB [orglq work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorglq( &n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Copy L to U, zeroing out above it
                    lapackf77_dlacpy( "L", &m, &m, A, &lda, U, &ldu );
                    lapackf77_dlaset( "U", &m_1, &m_1, &c_zero, &c_zero, U(0,1), &ldu );
                    ie    = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U
                    // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
                    // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd( &m, &m, U, &ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Multiply right bidiagonalizing vectors in U by Q in VT
                    // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr( "P", "L", "T", &m, &n, &m, U, &ldu, &work[itaup], VT, &ldvt, &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
                    #endif
                    
                    // Generate left bidiagonalizing vectors in U
                    // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
                    // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgbr( "Q", &m, &m, &m, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                    #else
                    magma_dorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                    #endif
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in VT
                    // Workspace: need     M [e] + 4*M [bdsqr work]
                    lapackf77_dbdsqr( "U", &m, &n, &m, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
                }
            }                                                     //
        }                                                         //
        else {                                                    //
            // N < MNTHR
            // Path 10t (N > M, but not much larger)
            dgesvd_path = "10t";
            // Reduce to bidiagonal form without LQ decomposition
            ie    = 1;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            
            // Bidiagonalize A
            // Workspace: need   3*M [e, tauq, taup] + N        [gebrd work]
            // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [gebrd work]
            lwork2 = lwork - iwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd( &m, &n, A, &lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr );
            #else
            magma_dgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
            #endif
            
            if (want_uas) {                                       //
                // If left singular vectors desired in U, copy result to U
                // and generate left bidiagonalizing vectors in U
                // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
                // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
                lapackf77_dlacpy( "L", &m, &m, A, &lda, U, &ldu );
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "Q", &m, &m, &n, U, &ldu, &work[itauq], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaQ,   m,  m,  n, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_vas) {                                       //
                // If right singular vectors desired in VT, copy result to
                // VT and generate right bidiagonalizing vectors in VT
                // Workspace: need     3*M [e, tauq, taup] + NRVT     [orgbr work]
                // Workspace: prefer   3*M [e, tauq, taup] + NRVT*NB  [orgbr work]
                lapackf77_dlacpy( "U", &m, &n, A,  &lda, VT, &ldvt );
                if (want_va) {
                    nrvt = n;
                }
                else {
                    assert(want_vs);
                    nrvt = m;
                }
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "P", &nrvt, &n, &m, VT, &ldvt, &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaP,  nrvt,  n,  m, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_uo) {                                        //
                // If left singular vectors desired in A, generate left
                // bidiagonalizing vectors in A
                // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
                // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "Q", &m, &m, &n, A, &lda, &work[itauq], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaQ,   m,  m,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            if (want_vo) {                                        //
                // If right singular vectors desired in A, generate right
                // bidiagonalizing vectors in A
                // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
                // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dorgbr( "P", &m, &n, &m, A, &lda, &work[itaup], &work[iwork], &lwork2, &ierr );
                #else
                magma_dorgbr( MagmaP,   m,  n,  m, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
                #endif
            }                                                     //
            
            iwork = ie + m;
            if (want_uas || want_uo) {
                nru = m;
            }
            if (want_un) {
                nru = 0;
            }
            if (want_vas || want_vo) {
                ncvt = n;
            }
            if (want_vn) {
                ncvt = 0;
            }
            if (! want_uo && ! want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in VT
                // Workspace: need       M [e] + 4*M [bdsqr work]
                lapackf77_dbdsqr( "L", &m, &ncvt, &nru, &izero, s, &work[ie], VT, &ldvt, U, &ldu, dummy, &ione, &work[iwork], info );
            }
            else if (! want_uo && want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in A
                // Workspace: need       M [e] + 4*M [bdsqr work]
                lapackf77_dbdsqr( "L", &m, &ncvt, &nru, &izero, s, &work[ie], A, &lda, U, &ldu, dummy, &ione, &work[iwork], info );
            }
            else {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in A and computing right singular
                // vectors in VT
                // Workspace: need       M [e] + 4*M [bdsqr work]
                lapackf77_dbdsqr( "L", &m, &ncvt, &nru, &izero, s, &work[ie], VT, &ldvt, A, &lda, dummy, &ione, &work[iwork], info );
            }
        }                                                         //
    }                                                             //
    
    // If DBDSQR failed to converge, copy unconverged superdiagonals
    // to WORK( 2:MINMN )
    if (*info != 0) {
        if (ie > 2) {
            m_1 = minmn - 1;
            for (i = 1; i <= m_1; ++i) {
                work[i + 1] = work[i + ie - 1];
            }
        }
        if (ie < 2) {
            for (i = minmn - 1; i >= 1; --i) {
                work[i + 1] = work[i + ie - 1];
            }
        }
    }
    
    // Undo scaling if necessary
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl( "G", &izero, &izero, &bignum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
        if (*info != 0 && anrm > bignum) {
            m_1 = minmn - 1;
            lapackf77_dlascl( "G", &izero, &izero, &bignum, &anrm, &m_1, &ione, &work[2], &minmn, &ierr );
        }
        if (anrm < smlnum) {
            lapackf77_dlascl( "G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, s, &minmn, &ierr );
        }
        if (*info != 0 && anrm < smlnum) {
            m_1 = minmn - 1;
            lapackf77_dlascl( "G", &izero, &izero, &smlnum, &anrm, &m_1, &ione, &work[2], &minmn, &ierr );
        }
    }

    // Return optimal workspace in WORK[1] (Fortran index)
    work[1] = magma_dmake_lwork( maxwrk );
    
    return *info;
} // magma_dgesvd
