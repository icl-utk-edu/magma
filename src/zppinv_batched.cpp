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
    ZPPINV computes the inverse of a complex Hermitian positive definite
    matrix A that is stored in packed format. The packed format stores either
    the lower (uplo = 'L') or the upper (uplo = 'U') part of the matrix.

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

    @param[in,out]
    dAP_array    Array of pointers, dimension (batchCount).
             Each is a AP is COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle of the Hermitian matrix
             A, packed columnwise in a linear array.  The j-th column of A
             is stored in the array AP as follows:
                 if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
                 if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.

              On exit, if INFO = 0, the inverse of 'A' is written using the
              same storage format as A.

    @param[in,out]
    device_work  Workspace, allocated on device (GPU) memory.

    @param[in,out]
    lwork_device   INTEGER pointer
                   The size of the workspace (device_work) in bytes
                   - lwork_device[0] < 0: a workspace query is assumed, the routine
                     calculates the required amount of workspace and returns
                     it in lwork_device. The workspace itself is not referenced, and no
                     computations is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[out]
    dinfo_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading principal minor of order i of A
                  is not positive, so the factorization could not be completed,
                  and the inverse has not been computed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_posv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zppinv_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dAP_array,
    void* device_work, int64_t *device_lwork,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    // calculate workspace
    int64_t workspace_bytes = 0;

    if ( uplo != MagmaLower ) {
        arginfo = -1;
        printf("Only uplo = MagmaLower is currently supported\n");
    }
    else if ( n < 0 || n > 64)
        arginfo = -2;
    else if ( device_lwork[0] > 0 && device_lwork[0] < workspace_bytes)
        arginfo = -5;
    else if ( batchCount < 0 )
        arginfo = -7;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // check for workspace query
    if( device_lwork[0] < 0 ) {
        device_lwork[0] = workspace_bytes;
        return arginfo;
    }

    /* Quick return if possible */
    if ( (n == 0) || (batchCount == 0) ) {
        return arginfo;
    }

    arginfo = magma_zpptrf_batched( uplo, n, dAP_array, dinfo_array, batchCount, queue);
    if ( arginfo == MAGMA_SUCCESS ) {
        arginfo = magma_zpptri_v2_batched_small(n, dAP_array, batchCount, dinfo_array, queue );
    }

    return arginfo;
}
