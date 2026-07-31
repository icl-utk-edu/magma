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
    ZHEMM-Packed performs the in-place update:
            B = AP * B     side='left', or
            B = B  * AP    side='right'
    , where AP is a Hermitian positive definite matrix stored in packed
    format, and B is a general matrix that is stored in column-major format.
    The packed format stores either the lower (uplo = 'L') or
    the upper (uplo = 'U') part of AP.

    ** Only side = 'left' is currently supported
    ** Only uplo = 'L'    is currently supported

    ** This is the batch version of the operation, performing the computation
       on many independent matrices having the same dimensions.

    ** The routine currently supports a limited range of sizes for the matrix AP

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether each Hermitian matrix AP
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    B := AP*B
            SIDE = MagmaRight   B := B*AP
      **    Only MagmaLeft is supported.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
      **    Only MagmaLower is supported.

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of each matrix B.
            m >= 0.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of each matrix AP.
            n >= 0.

    @param[in]
    dAP_array    Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle part of each matrix AP,
             packed columnwise in a linear array.

             Unchanged nn exit.

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddb,n)
               - On entry, the input matrix B.
               - On exit, the updated matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,n).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_posv_batched
*******************************************************************************/
extern "C" void
magma_zhemm_packed_inplace_batched(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dAP_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    if ( side != MagmaLeft  ) {
        arginfo = -1;
        printf("Only side = MagmaLeft is currently supported\n");
    }
    else if ( uplo != MagmaLower ) {
        arginfo = -2;
        printf("Only uplo = MagmaLower is currently supported\n");
    }
    else if ( m < 0 || m > 64 ) {
        arginfo = -3;
    }
    else if ( n < 0 ) {
        arginfo = -4;
    }
    else if ( lddb < max(1,m) ) {
        arginfo = -7;
    }
    else if ( batchCount < 0 )
        arginfo = -8;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    /* Quick return if possible */
    if ( m == 0 || n == 0 || batchCount == 0 ) return;

    // special case
    if( side == MagmaLeft && n == 1 ) {
        magma_zhemv_packed_inplace_batched_small( uplo, m, dAP_array, dB_array, lddb, batchCount, queue );
    }
    else {
        magma_zhemm_packed_inplace_batched_small(side, uplo, m, n, dAP_array, dB_array, lddb, batchCount, queue );
    }
}
