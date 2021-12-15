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
    ZGETRF computes an LU factorization of a general M-by-N matrix A
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
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_vbatched_max_nocheck(
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i_, j_)  dA_array, i_, j_
#define ipiv_array(i_)    ipiv_array, i_

    magma_int_t arginfo = 0;
    magma_int_t *minmn;
    magma_imalloc(&minmn, batchCount);
    magma_ivec_min_vv( batchCount, m, n, minmn, queue);
    magma_memset(info_array, 0, batchCount*sizeof(magma_int_t));

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t nb, recnb, ib, i, pm;

    // TODO: tuning
    nb    = 128;
    recnb = 32;

    for (i = 0; i < max_minmn; i += nb) {
        ib = min(nb, max_minmn-i);
        pm = max_m-i;

        // panel
        arginfo = magma_zgetrf_recpanel_vbatched(
                    m, n, minmn,
                    pm, ib, ib, 0, recnb,
                    dA_array(i, i), ldda,
                    ipiv_array(i), NULL,
                    info_array, i, batchCount, queue);

        if (arginfo != 0 ) goto fin;

        // swap left
        magma_zlaswp_left_rowserial_vbatched(
                i,
                m, n, dA_array(i, 0), ldda,
                ipiv_array(i),
                0, ib,
                batchCount, queue);

        if ( (i + ib) < max_n){
            // swap right
            magma_zlaswp_right_rowserial_vbatched(
                    max_n-(i+ib),
                    m, n, dA_array(i,i+ib), ldda,
                    ipiv_array(i),
                    0, ib,
                    batchCount, queue);

            // trsm
            magmablas_ztrsm_vbatched_core(
                    MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                    ib, max_n-i-ib, m, n, MAGMA_Z_ONE,
                    dA_array(i, i), ldda,
                    dA_array(i, i+ib), ldda,
                    batchCount, queue );

            if ( (i + ib) < max_m){
                // gemm update
                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        max_m-i-ib, max_n-i-ib, ib,
                        m, n, minmn,
                        MAGMA_Z_NEG_ONE, dA_array(i+ib, i   ), ldda,
                                         dA_array(i   , i+ib), ldda,
                        MAGMA_Z_ONE,     dA_array(i+ib, i+ib), ldda,
                        batchCount, queue );
            } // end of  if ( (i + ib) < max_m)

        } // end of if ( (i + ib) < max_n)

        // adjust pivot
        adjust_ipiv_vbatched(ipiv_array(i), minmn, ib, i, batchCount, queue);

    }// end of for

fin:
    magma_queue_sync(queue);
    magma_free(minmn);

    return arginfo;

#undef dA_array
#undef ipiv_array
}
