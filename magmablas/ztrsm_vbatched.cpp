/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Ahmad Abdelfattah
*/

#include "cublas_v2.h"
#include "magma_internal.h"
#include "commonblas_z.h"

#define PRECISION_z

/******************************************************************************/
extern "C" void 
magmablas_ztrsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue)
{
    magma_int_t info = 0;
    info =  magma_trsm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue);
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_ztrsm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, alpha, 
            dA_array, ldda, 
            dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue);
}

/******************************************************************************/
extern "C" void
magmablas_ztrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_ztrsm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, alpha, 
            dA_array, ldda, 
            dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue);
}

/***************************************************************************//**
    Purpose
    -------
    ztrsm solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or
        X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_ztrsm with flag,
    d_dinvA and dX workspaces as arguments.

    Arguments
    ----------
    @param[in]
    side    magma_side_t.
            On entry, side specifies whether op(A) appears on the left
            or right of X as follows:
      -     = MagmaLeft:       op(A)*X = alpha*B.
      -     = MagmaRight:      X*op(A) = alpha*B.

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    m       INTEGER array, dimension(batchCount + 1).
            On entry, each element M specifies the number of rows of 
            the corresponding B. M >= 0.

    @param[in]
    n       INTEGER array, dimension(batchCount + 1).
            On entry, each element N specifies the number of columns of 
            the corresponding B. N >= 0.

    @param[in]
    alpha   COMPLEX_16.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of dimension ( LDDA, k ), where k is M
             when side = MagmaLeft and is N when side = MagmaRight.
             Before entry with uplo = MagmaUpper, the leading k by k
             upper triangular part of the array A must contain the upper
             triangular matrix and the strictly lower triangular part of
             A is not referenced.
             Before entry with uplo = MagmaLower, the leading k by k
             lower triangular part of the array A must contain the lower
             triangular matrix and the strictly upper triangular part of
             A is not referenced.
             Note that when diag = MagmaUnit, the diagonal elements of
             A are not referenced either, but are assumed to be unity.

    @param[in]
    ldda    INTEGER array, dimension(batchCount + 1).
            On entry, each element LDDA specifies the first dimension of each array A.
            When side = MagmaLeft,  LDDA >= max( 1, M ),
            when side = MagmaRight, LDDA >= max( 1, N ).

    @param[in,out]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of dimension ( LDDB, N ).
             Before entry, the leading M by N part of the array B must
             contain the right-hand side matrix B.
             \n
             On exit, the solution matrix X

    @param[in]
    lddb    INTEGER array, dimension(batchCount + 1).
            On entry, LDDB specifies the first dimension of each array B.
            lddb >= max( 1, M ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C" void
magmablas_ztrsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    magma_int_t info = 0;
    info =  magma_trsm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue);
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_ztrsm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, alpha, 
            dA_array, ldda, 
            dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue);
}
