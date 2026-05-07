/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
/***************************************************************************//**
    Purpose
    -------
    ZPPSV computes the solution to a complex system of linear equations
    A * X = B, where A is an N-by-N Hermitian positive definite matrix stored in
    packed format and X and B are N-by-NRHS matrices.

    The Cholesky decomposition is used to factor A as
        A = U**H * U,  if UPLO = 'U', or
        A = L * L**H,  if UPLO = 'L',

     where U is an upper triangular matrix and L is a lower triangular
     matrix.  The factored form of A is then used to solve the system of
     equations A * X = B.

    ** Only uplo = 'L' is currently supported

    ** This is the batch version of the operation, performing the computation
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
            The order of the matrix A.  n >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dAP_array    Array of pointers, dimension (batchCount).
             Each is a AP is COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle of the Hermitian matrix
             A, packed columnwise in a linear array.  The j-th column of A
             is stored in the array AP as follows:
                 if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
                 if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.

              On exit, if INFO = 0, the factor U or L from the Cholesky
              factorization A = U**H*U or A = L*L**H, in the same storage
              format as A.

    @param[in,out]
    dB_array  Array of pointers, dimension (batchCount).
              Each is a COMPLEX*16 array, dimension (lddb,nrhs)
              On entry, the N-by-NRHS right hand side matrix B.
              On exit, if INFO = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,n).

    @param[out]
    dinfo_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading principal minor of order i of A
                  is not positive, so the factorization could not be completed,
                  and the solution has not been computed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_posv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zppsv_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dAP_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    if ( uplo != MagmaLower ) {
        arginfo = -1;
        printf("Only uplo = MagmaLower is currently supported\n");
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( nrhs < 0 )
        arginfo = -3;
    else if ( lddb < max(1, n) )
        arginfo = -6;
    else if ( batchCount < 0 )
        arginfo = -7;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return arginfo;
    }

    arginfo = magma_zpptrf_batched( uplo, n, dAP_array, dinfo_array, batchCount, queue);
    if ( arginfo == MAGMA_SUCCESS ) {
        arginfo = magma_zpptrs_batched( uplo, n, nrhs, dAP_array, dB_array, lddb,  batchCount, queue );
    }

    return arginfo;
}
