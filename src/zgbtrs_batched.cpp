/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"


////////////////////////////////////////////////////////////////////////////////
magma_int_t
magma_zgbtrs_lower_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i, j)    dA_array, i, j
#define dB_array(i, j)    dB_array, i, j

    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( transA != MagmaNoTrans ) {
        arginfo = -1;
        printf("ERROR: Function %s only support transA = MagmaNoTrans\n", __func__);
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if (nrhs < 0)
        arginfo = -5;
    else if ( ldda < (kl+kv+1) )
        arginfo = -7;
    else if ( lddb < n)
        arginfo = -10;
    else if ( batchCount < 0 )
        arginfo = -12;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n == 0 || batchCount == 0 || nrhs == 0) return 0;


    magma_int_t gbtrs_lower_info = -1;
    gbtrs_lower_info = magmablas_zgbtrs_lower_blocked_batched(
                        n, kl, ku, nrhs,
                        dA_array, ldda, dipiv_array,
                        dB_array, lddb,
                        batchCount, queue );

    if( gbtrs_lower_info != 0) {
        // ref. impl.: apply inv(L) as a series of row interchanges and rank-1 updates
        for(magma_int_t j = 0; j < n; j++) {
            // swap
            magmablas_zgbtrs_swap_batched(nrhs, dB_array, lddb, dipiv_array, j, batchCount, queue);

            // geru
            magmablas_zgeru_batched_core(
                min(kl, n-j-1), nrhs,
                MAGMA_Z_NEG_ONE,
                dA_array(kv+1, j), ldda, 1,
                dB_array(j   , 0), lddb, lddb,
                dB_array(j+1 , 0), lddb,
                batchCount, queue );
        }

        gbtrs_lower_info = 0;
    }

    return gbtrs_lower_info;
#undef dA_array
#undef dB_array
}


////////////////////////////////////////////////////////////////////////////////
magma_int_t
magma_zgbtrs_upper_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i, j)    dA_array, i, j
#define dB_array(i, j)    dB_array, i, j

    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( transA != MagmaNoTrans ) {
        arginfo = -1;
        printf("ERROR: Function %s only support transA = MagmaNoTrans\n", __func__);
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if (nrhs < 0)
        arginfo = -5;
    else if ( ldda < (kl+kv+1) )
        arginfo = -7;
    else if ( lddb < n)
        arginfo = -10;
    else if ( batchCount < 0 )
        arginfo = -12;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n == 0 || batchCount == 0 || nrhs == 0) return 0;

    // solve for U, backward solve
    magma_int_t gbtrs_upper_info = -1;
    gbtrs_upper_info = magmablas_zgbtrs_upper_blocked_batched(
                        n, kl, ku, nrhs,
                        dA_array, ldda,
                        dB_array, lddb,
                        batchCount, queue );

    if( gbtrs_upper_info != 0 ) {
        // ref. impl.: apply inv(U) column-wise
        for(magma_int_t j = n-1; j >= 0; j--) {
            magmablas_zgbtrs_upper_columnwise_batched(
                n, kl, ku, nrhs, j,
                dA_array, ldda,
                dB_array, lddb,
                batchCount, queue );
        }
        gbtrs_upper_info = 0;
    }

    return gbtrs_upper_info;

#undef dA_array
#undef dB_array
}

/***************************************************************************//**
    Purpose
    -------
    ZGBTRS solves a system of linear equations
    A * X = B,  A**T * X = B,  or  A**H * X = B with a general band matrix A
    using the LU factorization computed by ZGBTRF.

    This is the batched version of the routine.
    Currently, only (A * X = B) is supported (no-trans only)

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t
            Specifies the form of the system of equations.
            Currently, only MagnaNoTrans is supported (A*X = B)

    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    kl      INTEGER
            The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    ku      INTEGER
            The number of superdiagonals within the band of A.  KL >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA_array    Array of pointers, dimension (batchCount).
                Each contains the details of the LU factorization of the band matrix A,
                as computed by ZGBTRF.  U is stored as an upper triangular band
                matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and
                the multipliers used during the factorization are stored in
                rows KL+KU+2 to 2*KL+KU+1.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= (2*KL+KU+1).

    @param[in]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[in,out]
    dB_array    Array of pointers, dimension (batchCount).
                Each is a COMPLEX*16 array, dimension (LDB,NRHS)
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  LDDB >= max(1, N).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrs_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i, j)    dA_array, i, j
#define dB_array(i, j)    dB_array, i, j

    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( transA != MagmaNoTrans ) {
        arginfo = -1;
        printf("ERROR: Function %s only support transA = MagmaNoTrans\n", __func__);
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if (nrhs < 0)
        arginfo = -5;
    else if ( ldda < (kl+kv+1) )
        arginfo = -7;
    else if ( lddb < n)
        arginfo = -10;
    else if ( batchCount < 0 )
        arginfo = -12;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n == 0 || batchCount == 0 || nrhs == 0) return 0;

    magma_zgbtrs_lower_batched(
        transA, n, kl, ku, nrhs,
        dA_array, ldda, dipiv_array,
        dB_array, lddb,
        info_array, batchCount, queue);

    magma_zgbtrs_upper_batched(
        transA, n, kl, ku, nrhs,
        dA_array, ldda, dipiv_array,
        dB_array, lddb,
        info_array,
        batchCount, queue);

    return arginfo;
}

/// @see magma_zgbtrs_batched. This is the (pointer + stride) interface of magma_zgbtrs_batched
extern "C" magma_int_t
magma_zgbtrs_batched_strided(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t *dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( transA != MagmaNoTrans ) {
        arginfo = -1;
        printf("ERROR: Function %s only support transA = MagmaNoTrans\n", __func__);
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if (nrhs < 0)
        arginfo = -5;
    else if ( ldda < (kl+kv+1) )
        arginfo = -7;
    else if (strideA < (ldda * n))
        arginfo = -8;
    else if (stride_piv < n)
        arginfo = -10;
    else if ( lddb < n)
        arginfo = -12;
    else if (strideB < lddb * nrhs)
        arginfo = -13;
    else if ( batchCount < 0 )
        arginfo = -15;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n == 0 || batchCount == 0 || nrhs == 0) return 0;

    magmaDoubleComplex** dA_array   = (magmaDoubleComplex**)queue->get_dAarray();
    magmaDoubleComplex** dB_array   = (magmaDoubleComplex**)queue->get_dAarray();
    magma_int_t**      dipiv_array  = (magma_int_t**)queue->get_dCarray();

    magma_int_t max_batchCount   = queue->get_maxBatch();
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t batch = min(max_batchCount, batchCount-i);
        magma_zset_pointer(dA_array,    (magmaDoubleComplex*)(dA + i * strideA), ldda, 0, 0, strideA,    batch, queue);
        magma_zset_pointer(dB_array,    (magmaDoubleComplex*)(dB + i * strideB), lddb, 0, 0, strideB,    batch, queue);
        magma_iset_pointer(dipiv_array, (magma_int_t*)(dipiv + i * stride_piv),     1, 0, 0, stride_piv, batch, queue);

        magma_zgbtrs_batched(
            transA, n, kl, ku, nrhs,
            dA_array, ldda, dipiv_array,
            dB_array, lddb, info_array, batchCount, queue);
    }

    return arginfo;
}
