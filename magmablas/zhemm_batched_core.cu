/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
       
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_z
#include "hemm_template_kernel_batched.cuh"

/******************************************************************************/
extern "C" void 
magmablas_zhemm_batched_core(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t lddc, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
        magma_int_t batchCount, magma_queue_t queue )
{
    if(side == MagmaLeft){
        hemm_template_batched<magmaDoubleComplex, ZHEMM_BATCHED_LEFT>(
            side, uplo, m, n, 
            dA_array, ldda,
            dB_array, lddb, 
            dC_array, lddc, alpha, beta, 
            roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue);
    }else{
        hemm_template_batched<magmaDoubleComplex, ZHEMM_BATCHED_RIGHT>(
            side, uplo, m, n, 
            dA_array, ldda,
            dB_array, lddb, 
            dC_array, lddc, alpha, beta, 
            roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue);
    }
}

/***************************************************************************//**
    Purpose
    -------
    ZHEMM performs one of the matrix-matrix operations

        C := alpha*A*B + beta*C,
    or
        C := alpha*B*A + beta*C,

    where alpha and beta are scalars, A is a Hermitian matrix, and
    B and C are m by n matrices.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether each Hermitian matrix A
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    C := alpha*A*B + beta*C,
            SIDE = MagmaRight   C := alpha*B*A + beta*C.


    @param[in]
    uplo    magma_uplo_t
            On entry, uplo specifies whether the upper or lower
            triangular part of each Hermitian matrix A is to be
            referenced as follows:

            uplo = MagmaUpper   Only the upper triangular part of the
                                Hermitian matrix is to be referenced.
            uplo = MagmaLower   Only the lower triangular part of the
                                Hermitian matrix is to be referenced.

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
    dA_array    Array of pointers, dimension(batchCount).
            Each is a COMPLEX*16 array A of DIMENSION ( ldda, ka ), where ka is
            m when side = MagmaLower and is n otherwise.
            Before entry with side = MagmaLeft, the m by m part of
            the array A must contain the Hermitian matrix, such that
            when uplo = MagmaUpper, the leading m by m upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading m by m lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Before entry with side = MagmaRight, the n by n part of
            the array A must contain the Hermitian matrix, such that
            when uplo = MagmaUpper, the leading n by n upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading n by n lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero.

    @param[in]
    ldda    INTEGER
            On entry, ldda specifies the first dimension of each A as declared
            in the calling (sub) program.
            When side = MagmaLower then ldda >= max( 1, m ),
            otherwise                   ldda >= max( 1, n ).

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
magmablas_zhemm_batched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t lddc, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if (uplo != MagmaLower && uplo != MagmaUpper ) {
        info = -2;
    } else if ( m < 0 ) {
        info = -3;
    } else if ( n < 0 ) {
        info = -4;
    } else if ( ldda < max(1,nrowa) ) {
        info = -7;
    } else if ( lddb < max(1,m) ) {
        info = -9;
    } else if (lddc < max(1,m)) {
        info = -12;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_zhemm_batched_core( 
            side, uplo, 
            m, n, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            0, 0, 0, 0, 0, 0, 
            batchCount, queue );
}
