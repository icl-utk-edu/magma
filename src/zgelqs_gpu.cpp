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
    ZGELQS solves the minimum-norm solution to an underdetermined system
           min || X ||   subject to   A*X = C
    using the LQ factorization A = L*Q computed by ZGELQF_GPU.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. N >= M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,m, as returned by
            ZGELQF_GPU in the first m rows of its array argument dA.
            The lower triangular part contains L.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA, LDDA >= max(1,M).

    @param[in]
    tau     COMPLEX_16 array, dimension (M)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_ZGELQF_GPU.

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS right hand side matrix C.
            On exit, the N-by-NRHS solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB. LDDB >= max(1,N).

    @param[out]
    hwork   (workspace) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK,
            LWORK >= NB * N,
            where NB is the blocksize given by magma_get_zgelqf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_gelqs
*******************************************************************************/
extern "C" magma_int_t
magma_zgelqs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex const *tau,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    /* Constants */
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magma_int_t nb = magma_get_zgelqf_nb( m, n );
    magma_int_t lwkopt = nb * n;
    bool lquery = (lwork == -1);

    hwork[0] = magma_zmake_lwork( lwkopt );

    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0 || n < m)
        *info = -2;
    else if (nrhs < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;
    else if (lddb < max(1,n))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    if (m == 0 || n == 0 || nrhs == 0) {
        hwork[0] = c_one;
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /* Step 1: Solve L * Y = B(1:m, :)
     * L is m x m lower triangular, stored in the lower triangle of dA.
     * B(1:m,:) is overwritten with Y. */
    magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                 m, nrhs,
                 c_one, dA, ldda,
                        dB, lddb, queue );

    /* Step 2: Zero out rows m..n-1 of dB */
    if (n > m) {
        magmablas_zlaset( MagmaFull, n - m, nrhs,
                          MAGMA_Z_ZERO, MAGMA_Z_ZERO,
                          dB + m, lddb, queue );
    }

    magma_queue_destroy( queue );

    /* Step 3: Apply Q**H to expand from m to n dimensions.
     * X = Q**H * [Y; 0] where Q is from the LQ factorization A = L*Q.
     * Q is n x n, so Q**H * [Y; 0] is n x nrhs. */
    magma_zunmlq_gpu( MagmaLeft, MagmaConjTrans,
                      n, nrhs, m,
                      dA, ldda, tau,
                      dB, lddb,
                      hwork, lwork, info );

    return *info;
}
