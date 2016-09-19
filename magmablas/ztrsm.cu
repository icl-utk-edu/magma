/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
*/
#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "ztrtri.cuh"  // get NB from ztrtri

/***************************************************************************//**
    Purpose
    -------
    ztrsm_outofplace solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or
        X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The matrix X is output.

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
    m       INTEGER.
            On entry, m specifies the number of rows of B. m >= 0.

    @param[in]
    n       INTEGER.
            On entry, n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha   COMPLEX_16.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA      COMPLEX_16 array of dimension ( ldda, k ), where k is m
            when side = MagmaLeft and is n when side = MagmaRight.
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
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A.
            When side = MagmaLeft,  ldda >= max( 1, m ),
            when side = MagmaRight, ldda >= max( 1, n ).

    @param[in]
    dB      COMPLEX_16 array of dimension ( lddb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B.
            On exit, contents in the leading m by n part are destroyed.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of B.
            lddb >= max( 1, m ).

    @param[out]
    dX      COMPLEX_16 array of dimension ( lddx, n ).
            On exit, it contains the m by n solution matrix X.

    @param[in]
    lddx    INTEGER.
            On entry, lddx specifies the first dimension of X.
            lddx >= max( 1, m ).

    @param[in]
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

    @param
    d_dinvA (workspace) on device.
            If side == MagmaLeft,  d_dinvA must be of size dinvA_length >= ceil(m/NB)*NB*NB,
            If side == MagmaRight, d_dinvA must be of size dinvA_length >= ceil(n/NB)*NB*NB,
            where NB = 128.

    @param[in]
    dinvA_length   INTEGER.
            On entry, dinvA_length specifies the size of d_dinvA.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsm
*******************************************************************************/
extern "C"
void magmablas_ztrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dX(i_, j_) (dX + (i_) + (j_)*lddx)
    #define d_dinvA(i_) (d_dinvA + (i_)*NB)

    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;

    magma_int_t i, jb;
    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    magma_int_t min_dinvA_length;
    if ( side == MagmaLeft ) {
        min_dinvA_length = magma_roundup( m, NB )*NB;
    }
    else {
        min_dinvA_length = magma_roundup( n, NB )*NB;
    }
    
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (dA == NULL) {
        info = -8;
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (dB == NULL) {
        info = -10;
    } else if (lddb < max(1,m)) {
        info = -11;
    } else if (dX == NULL) {
        info = -12;
    } else if (lddx < max(1,m)) {
        info = -13;
    } else if (d_dinvA == NULL) {
        info = -15;
    } else if (dinvA_length < min_dinvA_length) {
        info = -16;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (m == 0 || n == 0)
        return;

    if (side == MagmaLeft) {
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag( uplo, diag, m, dA, ldda, d_dinvA, queue );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // left, lower no-transpose
                // handle first block separately with alpha
                jb = min(NB, m);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(0), NB, dB, lddb, c_zero, dX, lddx, queue );
                if (NB < m) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, dA(NB,0), ldda, dX, lddx, alpha, dB(NB,0), lddb, queue );

                    // remaining blocks
                    for( i=NB; i < m; i += NB ) {
                        jb = min(m-i, NB);
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                        if (i+NB >= m)
                            break;
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, dA(i+NB,i), ldda, dX(i,0), lddx, c_one, dB(i+NB,0), lddb, queue );
                    }
                }
            }
            else {
                // left, upper no-transpose
                // handle first block separately with alpha
                jb = (m % NB == 0) ? NB : (m % NB);
                i = m-jb;
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                if (i-NB >= 0) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA(0,i), ldda, dX(i,0), lddx, alpha, dB, lddb, queue );

                    // remaining blocks
                    for( i=m-jb-NB; i >= 0; i -= NB ) {
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, NB, n, NB, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                        if (i-NB < 0)
                            break;
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, NB, c_neg_one, dA(0,i), ldda, dX(i,0), lddx, c_one, dB, lddb, queue );
                    }
                }
            }
        }
        else {  // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // left, lower transpose
                // handle first block separately with alpha
                jb = (m % NB == 0) ? NB : (m % NB);
                i = m-jb;
                magma_zgemm( transA, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                if (i-NB >= 0) {
                    magma_zgemm( transA, MagmaNoTrans, i, n, jb, c_neg_one, dA(i,0), ldda, dX(i,0), lddx, alpha, dB, lddb, queue );

                    // remaining blocks
                    for( i=m-jb-NB; i >= 0; i -= NB ) {
                        magma_zgemm( transA, MagmaNoTrans, NB, n, NB, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                        if (i-NB < 0)
                            break;
                        magma_zgemm( transA, MagmaNoTrans, i, n, NB, c_neg_one, dA(i,0), ldda, dX(i,0), lddx, c_one, dB, lddb, queue );
                    }
                }
            }
            else {
                // left, upper transpose
                // handle first block separately with alpha
                jb = min(NB, m);
                magma_zgemm( transA, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(0), NB, dB, lddb, c_zero, dX, lddx, queue );
                if (NB < m) {
                    magma_zgemm( transA, MagmaNoTrans, m-NB, n, NB, c_neg_one, dA(0,NB), ldda, dX, lddx, alpha, dB(NB,0), lddb, queue );

                    // remaining blocks
                    for( i=NB; i < m; i += NB ) {
                        jb = min(m-i, NB);
                        magma_zgemm( transA, MagmaNoTrans, jb, n, jb, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), lddx, queue );
                        if (i+NB >= m)
                            break;
                        magma_zgemm( transA, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, dA(i,i+NB), ldda, dX(i,0), lddx, c_one, dB(i+NB,0), lddb, queue );
                    }
                }
            }
        }
    }
    else {  // side == MagmaRight
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag( uplo, diag, n, dA, ldda, d_dinvA, queue );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // right, lower no-transpose
                // handle first block separately with alpha
                jb = (n % NB == 0) ? NB : (n % NB);
                i = n-jb;
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                if (i-NB >= 0) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, i, jb, c_neg_one, dX(0,i), lddx, dA(i,0), ldda, alpha, dB, lddb, queue );

                    // remaining blocks
                    for( i=n-jb-NB; i >= 0; i -= NB ) {
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, NB, NB, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                        if (i-NB < 0)
                            break;
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, i, NB, c_neg_one, dX(0,i), lddx, dA(i,0), ldda, c_one, dB, lddb, queue );
                    }
                }
            }
            else {
                // right, upper no-transpose
                // handle first block separately with alpha
                jb = min(NB, n);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB, lddb, d_dinvA(0), NB, c_zero, dX, lddx, queue );
                if (NB < n) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n-NB, NB, c_neg_one, dX, lddx, dA(0,NB), ldda, alpha, dB(0,NB), lddb, queue );

                    // remaining blocks
                    for( i=NB; i < n; i += NB ) {
                        jb = min(NB, n-i);
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, jb, jb, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                        if (i+NB >= n)
                            break;
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n-i-NB, NB, c_neg_one, dX(0,i), lddx, dA(i,i+NB), ldda, c_one, dB(0,i+NB), lddb, queue );
                    }
                }
            }
        }
        else { // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // right, lower transpose
                // handle first block separately with alpha
                jb = min(NB, n);
                magma_zgemm( MagmaNoTrans, transA, m, jb, jb, alpha, dB, lddb, d_dinvA(0), NB, c_zero, dX, lddx, queue );
                if (NB < n) {
                    magma_zgemm( MagmaNoTrans, transA, m, n-NB, NB, c_neg_one, dX, lddx, dA(NB,0), ldda, alpha, dB(0,NB), lddb, queue );

                    // remaining blocks
                    for( i=NB; i < n; i += NB ) {
                        jb = min(NB, n-i);
                        magma_zgemm( MagmaNoTrans, transA, m, jb, jb, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                        if (i+NB >= n)
                            break;
                        magma_zgemm( MagmaNoTrans, transA, m, n-i-NB, NB, c_neg_one, dX(0,i), lddx, dA(NB+i,i), ldda, c_one, dB(0,i+NB), lddb, queue );
                    }
                }
            }
            else {
                // right, upper transpose
                // handle first block separately with alpha
                jb = (n % NB == 0) ? NB : (n % NB);
                i = n-jb;
                magma_zgemm( MagmaNoTrans, transA, m, jb, jb, alpha, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                if (i-NB >= 0) {
                    magma_zgemm( MagmaNoTrans, transA, m, i, jb, c_neg_one, dX(0,i), lddx, dA(0,i), ldda, alpha, dB, lddb, queue );

                    // remaining blocks
                    for( i=n-jb-NB; i >= 0; i -= NB ) {
                        magma_zgemm( MagmaNoTrans, transA, m, NB, NB, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), lddx, queue );
                        if (i-NB < 0)
                            break;
                        magma_zgemm( MagmaNoTrans, transA, m, i, NB, c_neg_one, dX(0,i), lddx, dA(0,i), ldda, c_one, dB, lddb, queue );
                    }
                }
            }
        }
    }
}


/***************************************************************************//**
    Similar to magmablas_ztrsm_outofplace(), but copies result dX back to dB,
    as in classical ztrsm interface.
    
    @see magmablas_ztrsm_outofplace
    @ingroup magma_trsm
*******************************************************************************/
extern "C"
void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue )
{
    magmablas_ztrsm_outofplace( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, queue );
    // copy X to B
    magmablas_zlacpy( MagmaFull, m, n, dX, lddx, dB, lddb, queue );
}


/***************************************************************************//**
    Similar to magmablas_ztrsm_outofplace(), but allocates dX and d_dinvA
    internally. This makes it a synchronous call, whereas
    magmablas_ztrsm_outofplace() and magmablas_ztrsm_work() are asynchronous.
    
    @see magmablas_ztrsm_work
    @ingroup magma_trsm
*******************************************************************************/
extern "C"
void magmablas_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (dA == NULL) {
        info = -8;
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (dB == NULL) {
        info = -10;
    } else if (lddb < max(1,m)) {
        info = -11;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    magmaDoubleComplex_ptr d_dinvA=NULL, dX=NULL;
    magma_int_t lddx = magma_roundup( m, 32 );
    magma_int_t size_x = lddx*n;
    magma_int_t dinvA_length;
    if ( side == MagmaLeft ) {
        dinvA_length = magma_roundup( m, NB )*NB;
    }
    else {
        dinvA_length = magma_roundup( n, NB )*NB;
    }

    magma_zmalloc( &d_dinvA, dinvA_length );
    magma_zmalloc( &dX, size_x );
    
    if ( d_dinvA == NULL || dX == NULL ) {
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        // continue to free
    }
    else {
        magmablas_zlaset( MagmaFull, dinvA_length, 1, MAGMA_Z_ZERO, MAGMA_Z_ZERO, d_dinvA, dinvA_length, queue );
        magmablas_zlaset( MagmaFull, m, n, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dX, lddx, queue );
        magmablas_ztrsm_work( side, uplo, transA, diag, m, n, alpha,
                                dA, ldda, dB, lddb, dX, lddx, 1, d_dinvA, dinvA_length, queue );
    }

    magma_free( d_dinvA );
    magma_free( dX );
}
