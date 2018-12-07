/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/*
 * Version 1 - LAPACK
 * Version 2 - MAGMA
 */
#define VERSION 2

/***************************************************************************//**
    Purpose
    -------
    ZGGLSE solves the linear equality-constrained least squares (LSE)
    problem:

            minimize || c - A*x ||_2   subject to   B*x = d

    where A is an M-by-N matrix, B is a P-by-N matrix, c is a given
    M-vector, and d is a given P-vector. It is assumed that
    P <= N <= M+P, and

             rank(B) = P and  rank( ( A ) ) = N.
                                  ( ( B ) )

    These conditions ensure that the LSE problem has a unique solution,
    which is obtained using a GRQ factorization of the matrices B and A.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrices A and B. N >= 0.

    @param[in]
    p       INTEGER
            The number of rows of the matrix B. 0 <= P <= N <= M+P.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, A is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    B       COMPLEX_16 array, dimension (LDB,N)
            On entry, the P-by-N matrix B.
            On exit, B is destroyed.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B. LDB >= max(1,P).

    @param[in,out]
    c       COMPLEX_16 array, dimension (M)
            On entry, C contains the right hand side vector for the
            least squares part of the LSE problem.
            On exit, the residual sum of squares for the solution
            is given by the sum of squares of elements N-P+1 to M of
            vector C.

    @param[in,out]
    d       COMPLEX_16 array, dimension (P)
            On entry, D contains the right hand side vector for the
            constrained equation.
            On exit, D is destroyed.

    @param[out]
    x       COMPLEX_16 array, dimension (N)
            On exit, x is the solution of the LSE problem.

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK. LWORK >= max(1,M+N+P).
            For optimum performance LWORK >= P+min(M,N)+max(M,N)*NB,
            where NB is an upper bound for the optimal blocksizes for
            ZGEQRF, CGERQF, ZUNMQR and CUNMRQ.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @ingroup magma_gglse
*******************************************************************************/
extern "C" magma_int_t
magma_zgglse(magma_int_t m, magma_int_t n, magma_int_t p,
             magmaDoubleComplex *A, magma_int_t lda,
             magmaDoubleComplex *B, magma_int_t ldb,
             magmaDoubleComplex *c, magmaDoubleComplex *d, magmaDoubleComplex *x,
             magmaDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info)
{
    #define  A(i_,j_)  (A + (i_) + (j_)*lda)
    #define  B(i_,j_)  (B + (i_) + (j_)*ldb)

    magmaDoubleComplex c_b1 = MAGMA_Z_ONE;
    magma_int_t one = 1;

    magma_int_t i__1, i__2;
    magmaDoubleComplex mone = MAGMA_Z_MAKE( -1., 0.);

    magma_int_t lopt, mn, nr, lwkopt;

    *info = 0;
    mn = min(m,n);
    magma_int_t nb = magma_get_zgeqrf_nb( m, n );
    lwkopt = p + mn + max(m,n) * nb;
    work[0] = magma_zmake_lwork( lwkopt );
    bool lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (p < 0 || p > n || p < n-m) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    } else if (ldb < max(1,p)) {
        *info = -7;
    } else /* if(complicated condition) */ {
        if (lwork < max(lwkopt,m+n+p) && ! lquery) {
            *info = -12;
        }
    }
    if (*info != 0) {
        magma_xerbla(__func__, -(*info) );
        return *info;
    } else if (lquery) {
        return *info;
    }

    /*     Quick return if possible */
    if (n == 0) {
        return *info;
    }

    /*     Compute the GRQ factorization of matrices B and A:

              B*Q' = (  0  T12 ) P   Z'*A*Q' = ( R11 R12 ) N-P
                       N-P  P                  (  0  R22 ) M+P-N
                                                 N-P  P

       where T12 and R11 are upper triangular, and Q and Z are
       unitary. */
    i__1 = lwork - p - mn;
    #if VERSION == 1
        lapackf77_zggrqf(&p, &m, &n, B(0,0), &ldb, work, A(0,0), &lda,
                         &work[p], &work[p+mn], &i__1, info);
    #else
        magma_zggrqf(p, m, n, B(0,0), ldb, work, A(0,0), lda,
                     &work[p], &work[p+mn], i__1, info);
    #endif
    lopt = (magma_int_t)MAGMA_Z_REAL( work[p+mn] );

    /*     Update c = Z'*c = ( c1 ) N-P
           ( c2 ) M+P-N */
    i__1 = max(1,m);
    i__2 = lwork - p - mn;
    #if VERSION == 1
        lapackf77_zunmqr(MagmaLeftStr, Magma_ConjTransStr, &m, &one, &mn, A(0,0), &lda,
                         &work[p], c, &i__1, &work[p+mn], &i__2, info);
    #else
        magma_zunmqr(MagmaLeft, Magma_ConjTrans, m, one, mn, A(0,0), lda,
                      &work[p], c, i__1, &work[p+mn], i__2, info);
    #endif

    /* Computing MAX */
    i__1 = lopt, i__2 = (magma_int_t)MAGMA_Z_REAL( work[p+mn] );
    lopt = max(i__1,i__2);

    /*     Solve T12*x2 = d for x2 */
    blasf77_ztrsv("Upper", "No transpose", "Non unit", &p, B(0, n-p), &ldb, d, &one);

    /*     Update c1 */
    i__1 = n - p;
    blasf77_zgemv("No transpose", &i__1, &p, &mone, A(0,n-p), &lda, d, &one, &c_b1, c, &one);

    /*     Sovle R11*x1 = c1 for x1 */
    i__1 = n - p;
    blasf77_ztrsv("Upper", "No transpose", "Non unit", &i__1, A(0,0), &lda, c, &one);

    /*     Put the solutions in X */
    i__1 = n - p;
    blasf77_zcopy(&i__1, c, &one, x, &one);
    blasf77_zcopy(&p, d, &one, &x[n-p], &one);

    /*     Compute the residual vector: */
    if (m < n) {
        nr = m + p - n;
        i__1 = n - m;
        blasf77_zgemv("No transpose", &nr, &i__1, &mone, A(n-p,m),
                      &lda, &d[nr], &one, &c_b1, &c[n-p], &one);
    } else {
        nr = p;
    }
    blasf77_ztrmv("Upper", "No transpose", "Non unit", &nr, A(n-p,n-p), &lda, d, &one);
    blasf77_zaxpy(&nr, &mone, d, &one, &c[n-p], &one);

    /*     Backward transformation x = Q'*x */
    i__1 = lwork - p - mn;
    #if VERSION == 1
        lapackf77_zunmrq(MagmaLeftStr, Magma_ConjTransStr, &n, &one, &p, B(0,0), &ldb,
                         work, x, &n, &work[p+mn], &i__1, info);
    #else
        magma_zunmrq(MagmaLeft, Magma_ConjTrans, n, one, p, B(0,0), ldb,
                     work, x, n, &work[p+mn], i__1, info);
    #endif

    /* Computing MAX */
    i__1 = p + mn + max(lopt, (magma_int_t) MAGMA_Z_REAL( work[p+mn] ) );
    work[0] = magma_zmake_lwork( i__1 );

    return *info;
} /* magma_zgglse */

