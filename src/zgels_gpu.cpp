/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZGELS solves the least squares problem
           min || A*X - C ||    (overdetermined, m >= n, via QR)
    or the minimum-norm problem
           min || X ||  subject to  A*X = C    (underdetermined, m < n, via LQ)
    using QR or LQ factorization of A.


    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:   the linear system involves A.
            Only TRANS=MagmaNoTrans is currently handled.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    @param[in,out]
    dA       COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, A is overwritten by details of its QR
            factorization as returned by ZGEQRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A, LDDA >= M.

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C (overdetermined) or M-by-NRHS
            right hand side (underdetermined).
            On exit, the N-by-NRHS solution matrix X (overdetermined) or
            the N-by-NRHS minimum-norm solution (underdetermined).

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB. LDDB >= max(1, max(M, N)).

    @param[out]
    hwork   (workspace) COMPLEX_16 array, dimension MAX(1,LWORK).
            On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK.
            If M >= N: LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_zgeqrf_nb( M, N ).
            If M < N:  LWORK >= NB * N,
            where NB is the blocksize given by magma_get_zgelqf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_gels
*******************************************************************************/
extern "C" magma_int_t
magma_zgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    magmaDoubleComplex_ptr dT;
    magmaDoubleComplex *tau;
    magma_int_t min_mn;
    magma_int_t nb, lwkopt;
    bool lquery = (lwork == -1);

    if (m >= n) {
        nb     = magma_get_zgeqrf_nb( m, n );
        lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    } else {
        nb     = magma_get_zgelqf_nb( m, n );
        lwkopt = nb * n;
    }

    hwork[0] = magma_zmake_lwork( lwkopt );

    *info = 0;
    if ( trans != MagmaNoTrans )
        *info = -1;
    else if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if (ldda < max(1,m))
        *info = -6;
    else if (lddb < max(1, max(m,n)))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    min_mn = min(m,n);
    if (min_mn == 0) {
        hwork[0] = MAGMA_Z_ONE;
        return *info;
    }

    magma_zmalloc_cpu( &tau, min_mn );
    if ( tau == NULL ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    if (m >= n) {
        /* === Overdetermined: QR path (existing) === */
        magma_int_t ldtwork = ( 2*min_mn + magma_roundup( n, 32 ) )*nb;
        if (nb < nrhs)
            ldtwork = ( 2*min_mn + magma_roundup( n, 32 ) )*nrhs;
        if (MAGMA_SUCCESS != magma_zmalloc( &dT, ldtwork )) {
            magma_free_cpu( tau );
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        magma_zgeqrf_gpu( m, n, dA, ldda, tau, dT, info );

        if ( *info == 0 ) {
            magma_zgeqrs_gpu( m, n, nrhs,
                              dA, ldda, tau, dT,
                              dB, lddb, hwork, lwork, info );
        }

        magma_free( dT );
    }
    else {
        /* === Underdetermined: LQ path === */
        /* 1. LQ factorize: A = L * Q */
        magma_zgelqf_gpu( m, n, dA, ldda, tau, hwork, lwork, info );

        if ( *info == 0 ) {
            /* 2. Solve L*Y = B, zero pad, apply Q**H */
            magma_zgelqs_gpu( m, n, nrhs,
                              dA, ldda, tau,
                              dB, lddb, hwork, lwork, info );
        }
    }

    magma_free_cpu( tau );
    return *info;
}
