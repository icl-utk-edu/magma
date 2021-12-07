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
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETF2 computes an LU factorization of a general M-by-N matrix A
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

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

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

    this is an internal routine that might have many assumption.


    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_vbatched(
    magma_int_t *m, magma_int_t *n,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i,j) dA_array, i, j
#define ipiv_array(i) ipiv_array, i

    magma_int_t arginfo = 0;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;

    magma_int_t panelj, step, j;

    #if 0
    for(j=0; j < max_minmn; j++) {
        // izamax
        magma_izamax_vbatched(max_m-j, m, n, dA_array(Ai+j, Aj+j), ldda, ipiv_array(Ai+j), info_array, j, gbstep, batchCount, queue);

        // zswap on the entire width
        magma_zswap_vbatched(max_n, m, n, dA_array(Ai+j, Aj), ldda, ipiv_array, batchCount, queue);

        // scal+ger
        magma_zscal_zgeru_vbatched( max_m, max_n, m, n, dA_array(Ai+j, Aj+j), ldda, info_array, j, gbstep, batchCount, queue);
    }
    #else
    magma_int_t *minmn_vec;
    magma_imalloc(&minmn_vec, batchCount);
    magma_ivec_min_vv( batchCount, m, n, minmn_vec, queue);

    magma_int_t nb = 8;
    for(j=0; j < max_minmn; j+=nb) {

        // panel (the swap is done on the entire width)
        magma_int_t ib = min(nb, max_minmn-j);
        for(magma_int_t jj = 0; jj < ib; jj++) {
            magma_int_t gbj = j+jj;
            // izamax
            magma_izamax_vbatched(max_m-gbj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, ipiv_array(Ai+gbj), info_array, gbj, gbstep, batchCount, queue);

            // zswap
            magma_zswap_vbatched(max_n, m, n, dA_array(Ai+gbj, Aj), ldda, ipiv_array, batchCount, queue);

            // scal+ger
            magma_zscal_zgeru_vbatched( max_m-gbj, ib-jj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, info_array, gbj, gbstep, batchCount, queue);
        }

        // trsm
        magmablas_ztrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            ib, max_n-j-ib, m, n, MAGMA_Z_ONE,
            dA_array(Ai+j, Aj+j   ), ldda,
            dA_array(Ai+j, Aj+j+ib), ldda, batchCount, queue );

        // gemm
        magmablas_zgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-j-ib, max_n-j-ib, ib,
            m, n, minmn_vec,
            MAGMA_Z_NEG_ONE, dA_array, Ai+j+ib, Aj+j,    ldda,
                             dA_array, Ai+j,    Aj+j+ib, ldda,
            MAGMA_Z_ONE,     dA_array, Ai+j+ib, Aj+j+ib, ldda,
            batchCount, queue );
    }

    magma_free( minmn_vec );
    #endif

    return 0;
#undef dA_array
#undef ipiv_array
}

