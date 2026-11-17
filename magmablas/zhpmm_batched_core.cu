/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
       @author Natalie Beams

*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_z
#include "hpmm_template_kernel_batched.cuh"

/******************************************************************************/
extern "C" void
magmablas_zhpmm_batched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dAP_array,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t lddc,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    if(side == MagmaLeft){
        hpmm_template_batched<magmaDoubleComplex, ZHEMM_BATCHED_LEFT>(
            side, uplo, m, n,
            dAP_array, nrowa,
            dB_array, lddb,
            dC_array, lddc, alpha, beta,
            roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue);
    }
}

/***************************************************************************//**
    Purpose
    -------
    ZHPMM performs one of the matrix-matrix operations

        C := alpha*AP*B + beta*C,
    or
        C := alpha*B*AP + beta*C,

    where alpha and beta are scalars, AP is a Hermitian matrix stored in a packed
    format (only upper or lower portion), and B and C are m by n matrices.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether each Hermitian matrix AP
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    C := alpha*AP*B + beta*C,
            SIDE = MagmaRight   C := alpha*B*AP + beta*C.

	    Currently, only MagmaLeft is supported.

    @param[in]
    uplo    magma_uplo_t
            On entry, uplo specifies whether the upper or lower
            triangular part of each Hermitian matrix AP is stored in the
	    packed format:

            uplo = MagmaUpper   Only the upper triangular part of the
                                Hermitian matrix is stored.
            uplo = MagmaLower   Only the lower triangular part of the
                                Hermitian matrix is stored.

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of each matrix C.
            m >= 0.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of each matrix C.
            n >= 0.

    @param[in]
    alpha   COMPLEX*16
            On entry, alpha specifies the scalar alpha.

    @param[in]
    dAP_array    Array of pointers, dimension(batchCount).
            Each is a COMPLEX*16 array AP of length ka*(ka+1)/2, where ka is
            m when side = MagmaLeft and is n otherwise.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero.

    @param[in]
    dB_array      Array of pointers, dimension(batchCount).
            Each is a COMPLEX*16 array B of DIMENSION ( lddb, n ).
            Before entry, the leading m by n part of the array B must
            contain the matrix B.

    @param[in]
    lddb    INTEGER
            On entry, lddb specifies the first dimension of B as declared
            in the calling (sub) program. LDDB >= max( 1, m ).

    @param[in]
    beta    COMPLEX*16
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC_array      Array of pointers, dimension(batchCount).
            Each is a COMPLEX*16 array C of DIMENSION ( lddc, n ).
            Before entry, the leading m by n part of the array C must
            contain the matrix C, except when beta is zero, in which
            case C need not be set on entry.
            On exit, the array C is overwritten by the m by n updated
            matrix.

    @param[in]
    lddc    INTEGER
            On entry, lddc specifies the first dimension of C as declared
            in the calling (sub) program. lddc >= max( 1, m ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_hemm_batched
*******************************************************************************/
extern "C" void
magmablas_zhpmm_batched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dAP_array,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    magma_int_t info = 0;
    if ( side != MagmaLeft ) {
        info = -1;
    } else if (uplo != MagmaLower && uplo != MagmaUpper ) {
        info = -2;
    } else if ( m < 0 ) {
        info = -3;
    } else if ( n < 0 ) {
        info = -4;
    } else if ( dB_array == dC_array && nrowa > 32 ) {
        info = -7;
    } else if ( lddb < max(1,m) ) {
        info = -8;
    } else if (lddc < max(1,m)) {
        info = -11;
    } else if (batchCount < 0) {
        info = -12;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    bool zhpmm_small = (
            side     == MagmaLeft    &&
            uplo     == MagmaLower   &&
            m        <= 32           &&
            alpha    == MAGMA_Z_ONE  &&
            beta     == MAGMA_Z_ZERO );

    // check for small zhpmm/zhpmv
    if( zhpmm_small ) {
        magma_zhpmm_small_batched(
            side, uplo, m, n,
            dAP_array,
            dB_array, lddb,
            dC_array, lddc,
            batchCount, queue);
        return;
    }

    // check for out-of-place hpmv
    if( side == MagmaLeft && n == 1 ) {
        magma_int_t ione = 1;
        magmablas_zhpmv_batched(
            uplo, m,
            alpha, dAP_array,
                   dB_array, ione,
            beta,  dC_array, ione,
            batchCount, queue );
        return;
    }

    magmablas_zhpmm_batched_core(
            side, uplo,
            m, n,
            alpha, dAP_array,
                   dB_array, lddb,
            beta,  dC_array, lddc,
            0, 0, 0, 0, 0, 0,
            batchCount, queue );
}
