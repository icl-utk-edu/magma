/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       @author Adrien REMY

       @precisions normal z -> s d c
*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZGESV solves a system of linear equations
        A * X = B
    where A is a general n-by-n matrix and X and B are n-by-nrhs matrices.
    The LU decomposition with no pivoting is used to factor A as
        A = L * U,
    where L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  nrhs >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (ldda,n).
            On entry, the n-by-n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  ldda >= max(1,n).

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (lddb,nrhs)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  lddb >= max(1,n).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_gesv_nopiv
*******************************************************************************/
extern "C" magma_int_t
magma_zgesv_nopiv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info )
{
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    } else if (lddb < max(1,n)) {
        *info = -6;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_zgetrf_nopiv_gpu( n, n, dA, ldda, info );
    if ( *info == MAGMA_SUCCESS ) {
        magma_zgetrs_nopiv_gpu( MagmaNoTrans, n, nrhs, dA, ldda, dB, lddb, info );
    }
    
    return *info;
}

extern "C" magma_int_t
magma_zgesv_nopiv_gpu_async(
        magma_int_t n, magma_int_t nrhs,
        magmaDoubleComplex_ptr dA, magma_int_t ldda,
        magmaDoubleComplex_ptr dB, magma_int_t lddb,
        magma_int_t *info, magma_queue_t queue )
{
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    } else if (lddb < max(1,n)) {
        *info = -6;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_zgetrf_nopiv_gpu_async( n, n, dA, ldda, info, queue );
    if ( *info == MAGMA_SUCCESS ) {
        magma_zgetrs_nopiv_gpu_async( MagmaNoTrans, n, nrhs, dA, ldda, dB, lddb, info, queue );
    }

    return *info;
}
