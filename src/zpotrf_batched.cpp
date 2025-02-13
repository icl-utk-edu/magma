/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    - This is the block version of the algorithm, calling Level 3 BLAS.
    - This is the fixed size batched version of the operation.
    - This is an expert interface that exposes the blocking sizes used internally
      during the factorization.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in]
    nb      INTEGER
            The blocking size used in the main loop of the factorization.  nb >= 0.

    @param[in]
    recnb   INTEGER
            The blocking size used in the recursive panel factorization.  0 <= recnb <= nb.

    @param[in,out]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             On entry, each pointer is a Hermitian matrix dA.
             If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of dA contains the upper
             triangular part of the matrix dA, and the strictly lower
             triangular part of dA is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of dA contains the lower
             triangular part of the matrix dA, and the strictly upper
             triangular part of dA is not referenced.
    \n
             On exit, if corresponding entry in info_array = 0,
             each pointer is the factor U or L from the Cholesky
             factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_potrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_expert_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t recnb,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)  dA_array, i, j

    magma_int_t arginfo = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (n < 0) {
        arginfo = -2;
    } else if (ldda < max(1,n)) {
        arginfo = -4;
    }
    else if(batchCount < 0) {
        arginfo = -6;
    }

    if (uplo == MagmaUpper) {
        printf("Upper side is not currently implemented\n");
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (n == 0 || batchCount  == 0) {
        return arginfo;
    }


    magma_int_t j, ib;
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.0, 0);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.0, 0);
    magma_device_t cdev;
    magma_getdevice( &cdev );

    for (j = 0; j < n; j += nb) {
        ib = min(nb, n-j);

        // update panel
        if( j > 0 ) {
            magma_zgemm_batched_core(
                MagmaNoTrans, MagmaConjTrans,
                n-j, ib, j,
                c_neg_one, dAarray(j, 0), ldda,
                           dAarray(j, 0), ldda,
                c_one,     dAarray(j, j), ldda,
                batchCount, queue );
        }

        //  panel factorization
        arginfo = magma_zpotrf_recpanel_batched(
                            uplo, n-j, ib, recnb,
                            dAarray(j, j), ldda,
                            info_array, j, batchCount, queue);
    }

    return arginfo;

#undef dAarray
}

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.
    This is the fixed size batched version of the operation.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             On entry, each pointer is a Hermitian matrix dA.
             If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of dA contains the upper
             triangular part of the matrix dA, and the strictly lower
             triangular part of dA is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of dA contains the lower
             triangular part of the matrix dA, and the strictly upper
             triangular part of dA is not referenced.
    \n
             On exit, if corresponding entry in info_array = 0,
             each pointer is the factor U or L from the Cholesky
             factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_potrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_memset(info_array, 0, batchCount*sizeof(magma_int_t));
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (n < 0) {
        arginfo = -2;
    } else if (ldda < max(1,n)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }


    magma_int_t nb, recnb;
    magma_get_zpotrf_batched_nbparam(n, &nb, &recnb);

    magma_int_t crossover = magma_get_zpotrf_batched_crossover();
    if (n > crossover){
        arginfo = magma_zpotrf_expert_batched(uplo, n, nb, recnb, dA_array, ldda, info_array, batchCount, queue);
    }
    else{
            arginfo = magma_zpotrf_lpout_batched(uplo, n, dA_array, 0, 0, ldda, 0, info_array, batchCount, queue);
    }

    return arginfo;
}
