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

#define ZGETRF2_VBATCHED_PAR_SWAP

extern "C" magma_int_t
magma_zgetrf_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **dipiv_array, magma_int_t** dpivinfo_array,
        magma_int_t *info_array, magma_int_t batchCount,
        magma_queue_t queue)
{
#define dA_array(i_, j_)  dA_array, i_, j_
#define dipiv_array(i_)    dipiv_array, i_

    magma_int_t arginfo = 0;
    magma_int_t ib, i, pm;

    // try a fused kernel for small sizes
    if( max_m <= 32 && max_n <= 32) {
        arginfo = magma_zgetf2_fused_vbatched(
                    max_m, max_n, max_minmn, max_mxn,
                    m, n,
                    dA_array, 0, 0, ldda,
                    dipiv_array, 0,
                    info_array, batchCount, queue );

        if(arginfo == 0) return arginfo;
    }

    for (i = 0; i < max_minmn; i += nb) {
        ib = min(nb, max_minmn-i);
        pm = max_m-i;

        // panel
        arginfo = magma_zgetrf_recpanel_vbatched(
                    m, n, minmn,
                    pm, ib, ib, max_mxn, recnb,
                    dA_array(i, i), ldda,
                    dipiv_array(i), dpivinfo_array,
                    info_array, i, batchCount, queue);

        if (arginfo != 0 ) goto fin;

        // swap left
        #ifdef ZGETRF2_VBATCHED_PAR_SWAP
        setup_pivinfo_vbatched(dpivinfo_array, i, dipiv_array, i, m, n, max_m-i, ib, batchCount, queue);
        magma_zlaswp_left_rowparallel_vbatched(
            i,
            m, n, dA_array(i, 0), ldda,
            0, ib,
            dpivinfo_array, i,
            batchCount, queue);
        #else
        magma_zlaswp_left_rowserial_vbatched(
                i,
                m, n, dA_array(i, 0), ldda,
                dipiv_array(i),
                0, ib,
                batchCount, queue);
        #endif

        if ( (i + ib) < max_n){
            // swap right
            #ifdef ZGETRF2_VBATCHED_PAR_SWAP
            magma_zlaswp_right_rowparallel_vbatched(
                    max_n-(i+ib),
                    m, n, dA_array(i,i+ib), ldda,
                    0, ib,
                    dpivinfo_array, i,
                    batchCount, queue);
            #else
            magma_zlaswp_right_rowserial_vbatched(
                    max_n-(i+ib),
                    m, n, dA_array(i,i+ib), ldda,
                    dipiv_array(i),
                    0, ib,
                    batchCount, queue);
            #endif

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
        adjust_ipiv_vbatched(dipiv_array(i), minmn, ib, i, batchCount, queue);

    }// end of for

    return arginfo;

#undef dA_array
#undef dipiv_array
}

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

    This is the variable-size batched version, which factors batchCount matrices of
    different sizes in parallel. Each matrix is assumed to have its own size and leading
    dimension.

    Arguments
    ---------
    @param[in]
    M       Array of INTEGERs on the GPU, dimension (batchCount)
            Each is the number of rows of each matrix A.  M[i] >= 0.

    @param[in]
    N       Array of INTEGERs on the GPU, dimension (batchCount)
            Each is the number of columns of each matrix A.  N[i] >= 0.

    @param[in,out]
    dA_array    Array of pointers on the GPU, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA[i],N[i]).
            On entry, each pointer is an M[i]-by-N[i] matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    Array of INTEGERs on the GPU
            Each is the leading dimension of each array A.  LDDA[i] >= max(1,M[i]).

    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M[i],N[i]))
            The pivot indices; for 1 <= p <= min(M[i],N[i]), row p of the
            matrix was interchanged with row IPIV(p).

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
magma_zgetrf_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    // error checker needs 3 integers, while the setup kernel requires 4
    // so allocate 4
    const magma_int_t stats_length = 4;

    magma_int_t arginfo = 0, hstats[stats_length];
    magma_int_t *minmn, *stats, *pivinfo;
    magma_int_t **dpivinfo_array;
    magma_imalloc(&stats, stats_length);    // max_m, max_n, max_minmn, max_mxn

    // the checker requires that stats contains at least 3 integers
    arginfo = magma_getrf_vbatched_checker( m, n, ldda, stats, batchCount, queue );

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
    }
    else {
        // min_mn
        magma_imalloc(&minmn, batchCount);
        magma_ivec_min_vv( batchCount, m, n, minmn, queue);
        magma_memset(info_array, 0, batchCount*sizeof(magma_int_t));

        // collect stats (requires at least 4 integers)
        magma_getrf_vbatched_setup( m, n, stats, batchCount, queue );
        magma_igetvector(stats_length, stats, 1, hstats, 1, queue);
        const magma_int_t max_m     = hstats[0];
        const magma_int_t max_n     = hstats[1];
        const magma_int_t max_minmn = hstats[2];
        const magma_int_t max_mxn   = hstats[3];

        // pivinfo
        magma_imalloc(&pivinfo, max_m * batchCount);
        magma_malloc((void**)&dpivinfo_array, batchCount * sizeof(magma_int_t*));
        magma_iset_pointer(dpivinfo_array, pivinfo, 1, 0, 0, max_m, batchCount, queue );

        magma_int_t nb, recnb;
        magma_get_zgetrf_vbatched_nbparam(max_m, max_n, &nb, &recnb);
        arginfo = magma_zgetrf_vbatched_max_nocheck(
                    m, n, minmn,
                    max_m, max_n, max_minmn, max_mxn,
                    nb, recnb,
                    dA_array, ldda,
                    dipiv_array, dpivinfo_array, info_array,
                    batchCount, queue);

        magma_queue_sync(queue);
        magma_free(minmn);
        magma_free(pivinfo);
        magma_free(dpivinfo_array);
    }

    magma_free(stats);
    return arginfo;
}
