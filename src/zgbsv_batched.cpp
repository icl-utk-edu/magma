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

/***************************************************************************//**
    Purpose
    -------
    ZGBSV computes the solution to a system of linear equations
    A * X = B, where A is a band matrix of order N with KL subdiagonals
    and KU superdiagonals, and X and B are N-by-NRHS matrices.

    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as A = L * U, where L is a product of permutation
    and unit lower triangular matrices with KL subdiagonals, and U is
    upper triangular with KL+KU superdiagonals.  The factored form of A
    is then used to solve the system of equations A * X = B.

    This is the batched version of the routine.

    Arguments
    ---------
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
            > 0:  if INFO = i, U(i,i) is exactly zero.  The factorization
                  has been completed, but the factor U is exactly
                  singular, and the solution has not been computed.

    @param[in,out]
    device_work  Workspace, allocated on device memory.

    @param[in,out]
    lwork        INTEGER pointer
                 The size of the workspace (device_work) in bytes
                 - lwork[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork. The workspace is not referenced, and no
                   computation is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbsv_batched_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( n < 0 )
        arginfo = -1;
    else if ( kl < 0 )
        arginfo = -2;
    else if ( ku < 0 )
        arginfo = -3;
    else if (nrhs < 0)
        arginfo = -4;
    else if ( ldda < (kl+kv+1) )
        arginfo = -6;
    else if ( lddb < n)
        arginfo = -9;
    else if ( batchCount < 0 )
        arginfo = -13;

    // calculate the amount of workspace required
    magma_int_t gbsv_lwork = 0, gbtrf_lwork = 0, gbtrs_lwork = 0;

    // query gbtrf, set gbtrf_lwork to a negative value
    gbtrf_lwork = -1;
    magma_zgbtrf_batched_work( n, n, kl, ku, NULL, ldda, NULL, NULL, NULL, &gbtrf_lwork, batchCount, queue);

    gbsv_lwork = gbtrf_lwork + gbtrs_lwork;
    if( *lwork < 0) {
       // workspace query is assumed
       *lwork = gbsv_lwork;
       return 0;
    }

    if(lwork[0] < gbsv_lwork) {
        arginfo = -12;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(n == 0 || batchCount == 0) return 0;

    // try fused kernel first
    magma_int_t nb    = 8, nthreads = kl+1;
    magma_int_t ntcol = 1;
    magma_get_zgbtrf_batched_params(n, n, kl, ku, &nb, &nthreads);
    magma_int_t fused_info = -1;
    if( n <= 64 && nrhs == 1) {
        fused_info = magma_zgbsv_batched_fused_sm(
                n, kl, ku, nrhs,
                dA_array, ldda, dipiv_array,
                dB_array, lddb, info_array,
                nthreads, ntcol, batchCount, queue );
    }
    if(fused_info == 0) return fused_info;

    // factorization
    magma_zgbtrf_batched_work(
        n, n,
        kl, ku,
        dA_array, ldda,
        dipiv_array, info_array,
        device_work, lwork, batchCount, queue);

    // solve
    magma_zgbtrs_batched(
        MagmaNoTrans, n, kl, ku, nrhs,
        dA_array, ldda, dipiv_array,
        dB_array, lddb, info_array,
        batchCount, queue);

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
/// @see magma_zgbsv_batched_work.
/// This is the (pointer + stride) interface of magma_zgbsv_batched_work
extern "C" magma_int_t
magma_zgbsv_batched_strided_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( n < 0 )
        arginfo = -1;
    else if ( kl < 0 )
        arginfo = -2;
    else if ( ku < 0 )
        arginfo = -3;
    else if (nrhs < 0)
        arginfo = -4;
    else if ( ldda < (kl+kv+1) )
        arginfo = -6;
    else if (strideA < ldda*n )
        arginfo = -7;
    else if (stride_piv < n)
        arginfo = -9;
    else if ( lddb < n)
        arginfo = -11;
    else if (strideB < lddb*nrhs)
        arginfo = -12;
    else if ( batchCount < 0 )
        arginfo = -16;

    magmaDoubleComplex** dA_array   = (magmaDoubleComplex**)queue->get_dAarray();
    magmaDoubleComplex** dB_array   = (magmaDoubleComplex**)queue->get_dBarray();
    magma_int_t**      dipiv_array  = (magma_int_t**)queue->get_dCarray();

    // calculate the amount of workspace required
    magma_int_t gbsv_lwork = 0, gbtrf_lwork = 0, gbtrs_lwork = 0;

    // query gbtrf, set gbtrf_lwork to a negative value
    gbtrf_lwork = -1;
    magma_zgbtrf_batched_work( n, n, kl, ku, NULL, ldda, NULL, NULL, NULL, &gbtrf_lwork, batchCount, queue);

    gbsv_lwork = gbtrf_lwork + gbtrs_lwork;
    if( *lwork < 0) {
       // workspace query is assumed
       *lwork = gbsv_lwork;
       return 0;
    }

    if(lwork[0] < gbsv_lwork) {
        arginfo = -15;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(n == 0 || batchCount == 0) return 0;

    magma_int_t max_batchCount   = queue->get_maxBatch();
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t batch = min(max_batchCount, batchCount-i);
        magma_zset_pointer(dA_array,    (magmaDoubleComplex*)(dA + i * strideA), ldda, 0, 0, strideA,   batch, queue);
        magma_zset_pointer(dB_array,    (magmaDoubleComplex*)(dB + i * strideB), lddb, 0, 0, strideB,   batch, queue);
        magma_iset_pointer(dipiv_array, (magma_int_t*)(dipiv + i * stride_piv),        1, 0, 0, stride_piv, batch, queue);

        magma_zgbsv_batched_work(
            n, kl, ku, nrhs,
            dA_array, ldda, dipiv_array,
            dB_array, lddb,
            info_array + i, device_work, lwork, batch, queue);
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGBSV computes the solution to a system of linear equations
    A * X = B, where A is a band matrix of order N with KL subdiagonals
    and KU superdiagonals, and X and B are N-by-NRHS matrices.

    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as A = L * U, where L is a product of permutation
    and unit lower triangular matrices with KL subdiagonals, and U is
    upper triangular with KL+KU superdiagonals.  The factored form of A
    is then used to solve the system of equations A * X = B.

    This is the batched version of the routine.

    Arguments
    ---------
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
magma_zgbsv_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( n < 0 )
        arginfo = -1;
    else if ( kl < 0 )
        arginfo = -2;
    else if ( ku < 0 )
        arginfo = -3;
    else if (nrhs < 0)
        arginfo = -4;
    else if ( ldda < (kl+kv+1) )
        arginfo = -6;
    else if ( lddb < n)
        arginfo = -9;
    else if ( batchCount < 0 )
        arginfo = -11;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n == 0 || batchCount == 0) return 0;

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbsv_batched_work(n, kl, ku, nrhs, NULL, ldda, NULL, NULL, lddb, NULL, NULL, lwork, batchCount, queue);

    void* device_work = NULL;
    magma_malloc(&device_work, lwork[0]);

    magma_zgbsv_batched_work(
        n, kl, ku, nrhs,
        dA_array, ldda, dipiv_array,
        dB_array, lddb,
        info_array,
        device_work, lwork, batchCount, queue);

    magma_free( device_work );
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
/// @see magma_zgbsv_batched.
/// This is the (pointer + stride) interface of magma_zgbsv_batched
extern "C"
magma_int_t
magma_zgbsv_batched_strided(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if ( n < 0 )
        arginfo = -1;
    else if ( kl < 0 )
        arginfo = -2;
    else if ( ku < 0 )
        arginfo = -3;
    else if (nrhs < 0)
        arginfo = -4;
    else if ( ldda < (kl+kv+1) )
        arginfo = -6;
    else if (strideA < ldda*n )
        arginfo = -7;
    else if (stride_piv < n)
        arginfo = -9;
    else if ( lddb < n)
        arginfo = -11;
    else if (strideB < lddb*nrhs)
        arginfo = -12;
    else if ( batchCount < 0 )
        arginfo = -14;

    // quick return if possible
    if(n == 0 || batchCount == 0) return 0;

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbsv_batched_strided_work(
        n, kl, ku, nrhs,
        NULL, ldda, strideA,
        NULL, stride_piv,
        NULL, lddb, strideB, NULL, NULL, lwork,
        batchCount, queue);

    void* device_work = NULL;
    magma_malloc(&device_work, lwork[0]);

    magma_zgbsv_batched_strided_work(
        n, kl, ku, nrhs,
        dA, ldda, strideA, dipiv, stride_piv,
        dB, lddb, strideB, info_array,
        device_work, lwork, batchCount, queue);

    magma_free( device_work );
    return arginfo;

    return arginfo;
}
