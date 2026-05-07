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

/***************************************************************************//**
    Purpose
    -------
    ZPPTRF computes the Cholesky factorization of a complex Hermitian positive
    definite matrix A stored in packed format.

    The factorization has the form:
        A = U**H * U,    if UPLO = 'U', or
        A = L    * L**H, if UPLO = 'L',

    where U is an upper triangular matrix and L is lower triangular.

    ** Only uplo = 'L' is currently supported

    ** This is the batch version of the operation, performing the factorization
       on many independent matrices having the same dimensions.

    ** The routine currently supports a limited range of matrix sizes

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dAP_array  Array of pointers, dimension (batchCount).
             Each is COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle of the Hermitian matrix
             A, packed columnwise in a linear array. The j-th column of AP
             is stored in the array AP as follows:
               - if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j)      for 1<=i<=j;
               - if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.

             On exit, if INFO = 0, the triangular factor U or L from the
             Cholesky factorization A = U**H*U or A = L*L**H, in the same
             storage format as A.

    @param[out]
    info_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading principal minor of order i
                  is not positive, and the factorization could not be
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
magma_zpptrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dAP_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if ( uplo != MagmaLower) {
        printf("Only uplo = MagmaLower is currently supported\n");
        arginfo = -1;
    } else if (n < 0) {
        arginfo = -2;
    }
    else if(batchCount < 0) {
        arginfo = -5;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (n == 0 || batchCount  == 0) {
        return arginfo;
    }

    // hard-coded tuning for now
    if( n <= 8) {
        arginfo = magma_zpptrf_batched_small( uplo, n, dAP_array, info_array, batchCount, queue );
    }
    else {
        arginfo = magma_zpptf2_batched_small( uplo, n, dAP_array, info_array, batchCount, queue );
    }

    return arginfo;

}
