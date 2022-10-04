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
#include "commonblas_z.h"

#define PRECISION_z
/******************************************************************************/
extern "C" void 
magmablas_zhemm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue )
{
    magmablas_zhemm_vbatched_core( 
           side, uplo, m, n, 
           alpha, dA_array, ldda,
                  dB_array, lddb, 
           beta,  dC_array, lddc, 
           max_m, max_n, 
           0, 0, 0, 0, 0, 0, 0, 0, 
           batchCount, queue );
}

/******************************************************************************/
extern "C" void
magmablas_zhemm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info = magma_hemm_vbatched_checker(side, uplo, m, n, ldda, lddb, lddc, batchCount, queue );

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_zhemm_vbatched_max_nocheck(
            side, uplo, 
            m, n, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, max_m, max_n, 
            queue );
}

/******************************************************************************/
extern "C" void
magmablas_zhemm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue )
{
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );    // maybe not needed
    
    magmablas_zhemm_vbatched_max_nocheck(
            side, uplo, 
            m, n, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, max_m, max_n, queue );
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
    m       INTEGER array, dimension(batchCount + 1).
            On entry, each element M specifies the number of rows of each matrix C.
            M >= 0.

    @param[in]
    n       INTEGER array, dimension(batchCount + 1). 
            On entry, each element N specifies the number of columns of each matrix C.
            N >= 0.

    @param[in]
    alpha   COMPLEX*16
            On entry, alpha specifies the scalar alpha.

    @param[in]
    dA_array    Array of pointers, dimension(batchCount).
            Each is a COMPLEX*16 array A of DIMENSION ( LDDA, ka ), where ka is
            M when side = MagmaLower and is N otherwise.
            Before entry with side = MagmaLeft, the M by M part of
            the array A must contain the Hermitian matrix, such that
            when uplo = MagmaUpper, the leading M by M upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading M by M lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Before entry with side = MagmaRight, the N by N part of
            the array A must contain the Hermitian matrix, such that
            when uplo = MagmaUpper, the leading N by N upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading N by N lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero.

    @param[in]
    ldda    INTEGER array, dimension(batchCount + 1).
            On entry, each element LDDA specifies the first dimension of each A as declared
            in the calling (sub) program.
            When side = MagmaLower then LDDA >= max( 1, M ),
            otherwise                   LDDA >= max( 1, N ).

    @param[in]
    dB_array      Array of pointers, dimension(batchCount). 
            Each is a COMPLEX*16 array B of DIMENSION ( LDDB, N ).
            Before entry, the leading M by N part of the array B must
            contain the matrix B.

    @param[in]
    lddb    INTEGER array, dimension(batchCount + 1).
            On entry, each element LDDB specifies the first dimension of B as declared
            in the calling (sub) program. LDDB >= max( 1, M ).

    @param[in]
    beta    COMPLEX*16
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC_array      Array of pointers, dimension(batchCount). 
            Each is a COMPLEX*16 array C of DIMENSION ( LDDC, N ).
            Before entry, the leading M by N part of the array C must
            contain the matrix C, except when beta is zero, in which
            case C need not be set on entry.
            On exit, the array C is overwritten by the M by N updated
            matrix.

    @param[in]
    lddc    INTEGER array, dimension(batchCount + 1).
            On entry, each element LDDC specifies the first dimension of C as declared
            in the calling (sub) program. LDDC >= max( 1, M ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    

    @ingroup magma_hemm_batched
*******************************************************************************/

extern "C" void
magmablas_zhemm_vbatched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info = magma_hemm_vbatched_checker(side, uplo, m, n, ldda, lddb, lddc, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );    // maybe not needed

    magmablas_zhemm_vbatched_max_nocheck(
            side, uplo, 
            m, n, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, max_m, max_n, queue );
}

/******************************************************************************/
