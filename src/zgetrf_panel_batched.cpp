/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    This is an internal routine that might have many assumption.
    Documentation is not fully completed

    ZGETRF_PANEL computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in]
    min_recpnb   INTEGER.
                 Internal use. The recursive nb

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai     INTEGER
           Row offset for A.

    @param[in]
    aj     INTEGER
           Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dpivinfo_array  Array of pointers, dimension (batchCount), for internal use.

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb, 
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,  
    magma_int_t batchCount,  magma_queue_t queue)
{
#define dAarray(i,j)    dA_array, i, j
#define ipiv_array(i)   dipiv_array, i

    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ai < 0) {
        arginfo = -4;
    } else if (aj < 0 || aj != ai) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if (n <= min_recpnb) {
        magma_zgetf2_batched(m, n,
                dAarray(ai,aj), ldda,
                dipiv_array, dpivinfo_array, 
                info_array, gbstep, batchCount, queue);
    }
    else {
        magma_int_t n1 = n / 2;
        magma_int_t n2 = n - n1;

        // panel
        magma_zgetrf_recpanel_batched(
                m, n1, min_recpnb, 
                dAarray(ai, aj), ldda, 
                dipiv_array, dpivinfo_array, 
                info_array, gbstep, batchCount, queue);

        // swap right
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai), m, n1, batchCount, queue);
        magma_zlaswp_rowparallel_batched(
                n2, 
                dAarray(ai,aj+n1), ldda, 
                dAarray(ai,aj+n1), ldda, 
                0, n1, dpivinfo_array, 
                batchCount, queue);

        // trsm
        magmablas_ztrsm_recursive_batched(
                MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
                n1, n2, MAGMA_Z_ONE, 
                dAarray(ai,   aj), ldda, 
                dAarray(ai,aj+n1), ldda,  
                batchCount, queue );

        // gemm
        magma_zgemm_batched_core( 
                MagmaNoTrans, MagmaNoTrans, 
                m-n1, n2, n1, 
                MAGMA_Z_NEG_ONE, dAarray(ai+n1,    aj), ldda, 
                                 dAarray(ai   , aj+n1), ldda, 
                MAGMA_Z_ONE,     dAarray(ai+n1, aj+n1), ldda, 
                batchCount, queue );

        // panel 2
        //printf("panel 2\n");
        magma_zgetrf_recpanel_batched(
                m-n1, n2, min_recpnb, 
                dAarray(ai+n1, aj+n1), ldda, 
                dipiv_array, dpivinfo_array, 
                info_array, gbstep + n1, batchCount, queue);

        // swap left
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai+n1), m-n1, n2, batchCount, queue);
        adjust_ipiv_batched(ipiv_array(ai+n1), n2, n1, batchCount, queue);
        magma_zlaswp_rowparallel_batched(
                n1, 
                dAarray(ai+n1,aj), ldda, 
                dAarray(ai+n1,aj), ldda, 
                n1, n, dpivinfo_array, 
                batchCount, queue);
    }

    return 0;

    #undef dAarray
    #undef ipiv_array
}
