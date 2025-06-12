/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Wajih-Halim Boukaram
   @author Yang Liu
   @author Sherry Li

   @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

extern "C" magma_int_t
magma_zgetrf_nopiv_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double* dtol_array, double eps,
        magma_int_t *info_array, magma_int_t batchCount,
        magma_queue_t queue)
{
#define dA_array(i_, j_)  dA_array, i_, j_

    magma_int_t arginfo = 0;
    magma_int_t ib, i, pm;

    magma_memset_async(info_array, 0, batchCount*sizeof(magma_int_t), queue);

    // try a fused kernel for small sizes
    if( max_m <= 32 && max_n <= 32) {
        arginfo = magma_zgetf2_nopiv_fused_vbatched(
                    max_m, max_n, max_minmn, max_mxn,
                    m, n,
                    dA_array, 0, 0, ldda,
                    dtol_array, eps,
                    info_array, batchCount, queue );

        if(arginfo == 0) return arginfo;
    }

    for (i = 0; i < max_minmn; i += nb) {
        ib = min(nb, max_minmn-i);
        pm = max_m-i;

        // panel
        arginfo = magma_zgetrf_nopiv_recpanel_vbatched(
                    m, n, minmn,
                    pm, ib, ib, max_mxn, recnb,
                    dA_array(i, i), ldda,
                    dtol_array, eps,
                    info_array, i, batchCount, queue);

        if (arginfo != 0 ) return arginfo;

        if ( (i + ib) < max_n){
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

    }// end of for

    return arginfo;

#undef dA_array
}

/***************************************************************************//**
    Purpose
    -------
    ZGETRF NOPIV computes an LU factorization of a general M-by-N matrix A
    without pivoting. It replaces tiny pivots smaller than a specified tolerance
    by that tolerance

    The factorization has the form
        A = L * U
    where L is lower triangular with unit diagonal elements (lower trapezoidal
    if m > n), and U is upper triangular (upper trapezoidal if m < n).

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

    @param[in]
    MAX_M   INTEGER
            The maximum number of rows across the batch

    @param[in]
    MAX_N   INTEGER
            The maximum number of columns across the batch

    @param[in]
    MAX_MINMN INTEGER
              The maximum value of min(Mi, Ni) for i = 1, 2, ..., batchCount

    @param[in]
    MAX_MxN INTEGER
            The maximum value of the product (Mi x Ni) for i = 1, 2, ..., batchCount

    @param[in,out]
    dA_array    Array of pointers on the GPU, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA[i],N[i]).
            On entry, each pointer is an M[i]-by-N[i] matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    Array of INTEGERs on the GPU
            Each is the leading dimension of each array A.  LDDA[i] >= max(1,M[i]).

    @param[in]
    dtol_array  Array of DOUBLEs, dimension (batchCount), for corresponding matrices.
            Each is the tolerance that is compared to the diagonal element before
            the column is scaled by its inverse. If the value of the diagonal is less
            than the threshold, the diagonal is replaced by the threshold.
            If the array is set to NULL, then the threshold is set to the eps parameter

    @param[in]
    eps     DOUBLE
            The value to use for the tolerance for all matrices if the dtol_array is NULL

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations. If a tolerance array is specified
                  the value shows the number of times a tiny pivot was replaced

    @param[in]
    WORK        VOID pointer
                A workspace of size LWORK[0]

    @param[inout]
    LWORK       INTEGER pointer
                If lwork[0] < 0, a workspace query is assumed, and lwork[0] is
                overwritten by the required workspace size in bytes.
                Otherwise, lwork[0] is the size of work

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_nopiv_vbatched_max_nocheck_work(
        magma_int_t* m, magma_int_t* n,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array,
        void* work, magma_int_t* lwork,
        magma_int_t batchCount, magma_queue_t queue)
{
    // first calculate required workspace in bytes
    magma_int_t workspace_bytes = 0;
    workspace_bytes += batchCount * sizeof(magma_int_t);         // minmn array
    workspace_bytes  = magma_roundup(workspace_bytes, 128);      // multiple of 128 bytes

    if( *lwork < 0 ) {
        // workspace query is assumed
        *lwork = workspace_bytes;
        return 0;
    }

    // check lwork
    if( *lwork < workspace_bytes ) {
        printf("error in %s, not enough workspace (lwork = %lld, required = %lld)\n",
                __func__, (long long)(*lwork), (long long)workspace_bytes );
        return -12;    // lwork is not enough
    }

    // split workspace as needed by magma_zgetrf_nopiv_vbatched_max_nocheck
    magma_int_t* minmn           = (magma_int_t*)work;

    // init
    magma_ivec_min_vv( batchCount, m, n, minmn, queue);

    // blocking sizes
    magma_int_t nb, recnb;
    magma_get_zgetrf_vbatched_nbparam(max_m, max_n, &nb, &recnb);

    // call magma_zgetrf_nopiv_vbatched_max_nocheck
    return magma_zgetrf_nopiv_vbatched_max_nocheck(
                m, n, minmn,
                max_m, max_n, max_minmn, max_mxn,
                nb, recnb,
                dA_array, ldda,
                dtol_array, eps, info_array,
                batchCount, queue);
}

/***************************************************************************//**
    Purpose
    -------
    ZGETRF NOPIV computes an LU factorization of a general M-by-N matrix A
    without pivoting. It replaces tiny pivots smaller than a specified tolerance
    by that tolerance.

    The factorization has the form
        A = L * U
    where L is lower triangular with unit diagonal elements (lower trapezoidal
    if m > n), and U is upper triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is the variable-size batched version, which factors batchCount matrices of
    different sizes in parallel. Each matrix is assumed to have its own size and leading
    dimension.

    This is the expert version taking an extra parameter for the tolerance for diagonal
    elements. Small diagonal elements will be replaced by the specified tolerance preserving
    the sign and the info array will report the number of replacements. This is useful in the
    context of static pivoting used in sparse solvers such as SuperLU, where the tolerance would
    be the the norm of the matrix scaled by the machine epsilon for example.

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
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    Array of INTEGERs on the GPU
            Each is the leading dimension of each array A.  LDDA[i] >= max(1,M[i]).

    @param[in]
    dtol_array  Array of DOUBLEs, dimension (batchCount), for corresponding matrices.
            Each is the tolerance that is compared to the diagonal element before
            the column is scaled by its inverse. If the value of the diagonal is less
            than the threshold, the diagonal is replaced by the threshold.
            If the array is set to NULL, then the threshold is set to the eps parameter

    @param[in]
    eps     DOUBLE
            The value to use for the tolerance for all matrices if the dtol_array is NULL

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations. If a tolerance array is specified
                  the value shows the number of times a tiny pivot was replaced

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_nopiv_expert_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    // error checker needs 3 integers, while the setup kernel requires 4
    // so allocate 4
    const magma_int_t stats_length = 4;

    magma_int_t arginfo = 0, hstats[stats_length];
    magma_int_t *stats;
    magma_imalloc(&stats, stats_length);    // max_m, max_n, max_minmn, max_mxn

    // the checker requires that stats contains at least 3 integers
    arginfo = magma_getrf_vbatched_checker( m, n, ldda, stats, batchCount, queue );

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
    }
    else {
        void* device_work;
        magma_int_t lwork[1];

        // collect stats (requires at least 4 integers)
        magma_getrf_vbatched_setup( m, n, stats, batchCount, queue );
        magma_igetvector(stats_length, stats, 1, hstats, 1, queue);
        const magma_int_t max_m     = hstats[0];
        const magma_int_t max_n     = hstats[1];
        const magma_int_t max_minmn = hstats[2];
        const magma_int_t max_mxn   = hstats[3];

        // query workspace
        lwork[0] = -1;
        magma_zgetrf_nopiv_vbatched_max_nocheck_work(
            NULL, NULL,
            max_m, max_n, max_minmn, max_mxn,
            NULL, NULL, NULL, eps, NULL,
            NULL, lwork, batchCount, queue);

        // alloc workspace
        magma_malloc( (void**)&device_work, lwork[0] );

        arginfo = magma_zgetrf_nopiv_vbatched_max_nocheck_work(
                    m, n,
                    max_m, max_n, max_minmn, max_mxn,
                    dA_array, ldda,
                    dtol_array, eps, info_array,
                    device_work, lwork,
                    batchCount, queue);

        magma_queue_sync( queue );
        magma_free( device_work );
    }

    magma_free(stats);
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGETRF NOPIV computes an LU factorization of a general M-by-N matrix A
    without pivoting.

    The factorization has the form
        A = L * U
    where L is lower triangular with unit diagonal elements (lower trapezoidal
    if m > n), and U is upper triangular (upper trapezoidal if m < n).

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
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    Array of INTEGERs on the GPU
            Each is the leading dimension of each array A.  LDDA[i] >= max(1,M[i]).

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
magma_int_t
magma_zgetrf_nopiv_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue) {

        return magma_zgetrf_nopiv_expert_vbatched(
                m, n, dA_array, ldda, NULL, 0, info_array, batchCount, queue);
}