
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
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZPPTRS solves a system of linear equations A*X = B with a Hermitian positive
    definite matrix A in packed storage using the Cholesky
    factorization A = U**H * U or A = L * L**H computed by ZPPTRF.

    ** Only uplo = 'L' is currently supported

    ** This is the batch version of the operation, performing the computation
       on many independent problems having the same dimensions.

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

    @param[in]
    dAP_array    Array of pointers, dimension (batchCount).
             Each (input) COMPLEX*16 array, dimension (n*(n+1)/2)
             The triangular factor U or L from the Cholesky factorization
             A = U**H * U or A = L * L**H, packed columnwise in a linear array.
             The j-th column of U or L is stored in the array AP as follows:
               - if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j;
               - if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n.

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddb,nrhs)
               - On entry, the right hand side matrix B.
               - On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,n).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_potrs_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpptrs_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dAP_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if ( uplo != MagmaLower ) {
        printf("Only uplo = MagmaLower is currently supported\n");
        arginfo = -1;
    }
    else if ( n < 0 )
        arginfo = -2;
    else if ( nrhs < 0)
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

    arginfo = magma_zpptrs_batched_small(n, nrhs, dAP_array, dB_array, lddb, batchCount, queue );

    return arginfo;
}
