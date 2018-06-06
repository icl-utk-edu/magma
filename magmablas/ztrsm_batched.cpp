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
       @author Azzam Haidar
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

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
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

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
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of dimension ( ldda, k ), where k is m
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
            On entry, ldda specifies the first dimension of each array A.
            When side = MagmaLeft,  ldda >= max( 1, m ),
            when side = MagmaRight, ldda >= max( 1, n ).

    @param[in]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of dimension ( lddb, n ).
             Before entry, the leading m by n part of the array B must
             contain the right-hand side matrix B.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B.
            lddb >= max( 1, m ).

    @param[in,out]
    dX_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array X of dimension ( lddx, n ).
             On entry, should be set to 0
             On exit, the solution matrix X

    @param[in]
    lddx    INTEGER.
            On entry, lddx specifies the first dimension of each array X.
            lddx >= max( 1, m ).

    @param
    dinvA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array dinvA, a workspace on device.
            If side == MagmaLeft,  dinvA must be of size >= ceil(m/ZTRTRI_BATCHED_NB)*ZTRTRI_BATCHED_NB*ZTRTRI_BATCHED_NB
            If side == MagmaRight, dinvA must be of size >= ceil(n/ZTRTRI_BATCHED_NB)*ZTRTRI_BATCHED_NB*ZTRTRI_BATCHED_NB

    @param[in]
    dinvA_length    INTEGER
                   The size of each workspace matrix dinvA
                   
    @param
    dA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dB_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dX_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dinvA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param[in]
    resetozero INTEGER
               Used internally by ZTRTRI_DIAG routine
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C"
void magmablas_ztrsm_inv_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx, 
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, 
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dX(i_, j_) (dX + (i_) + (j_)*m)
    #define d_dinvA(i_) (d_dinvA + (i_)*ZTRTRI_BATCHED_NB)
    */
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;

    magma_int_t i, jb;
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
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    }
    magma_int_t size_dinvA;
    if ( side == MagmaLeft ) {
        size_dinvA = magma_roundup( m, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    }
    else {
        size_dinvA = magma_roundup( n, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    }
    if (dinvA_length < size_dinvA) info = -19;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (m == 0 || n == 0)
        return;

    magma_zdisplace_pointers(dA_displ,       dA_array,    ldda,    0, 0, batchCount, queue); 
    magma_zdisplace_pointers(dB_displ,       dB_array,    lddb,    0, 0, batchCount, queue); 
    magma_zdisplace_pointers(dX_displ,       dX_array,    lddx,    0, 0, batchCount, queue); 
    magma_zdisplace_pointers(dinvA_displ, dinvA_array,  ZTRTRI_BATCHED_NB,    0, 0, batchCount, queue); 

    if (side == MagmaLeft) {
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag_batched( uplo, diag, m, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // left, lower no-transpose
                // handle first block seperately with alpha
                jb = min(ZTRTRI_BATCHED_NB, m);                
                magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (ZTRTRI_BATCHED_NB < m) {
                    magma_zdisplace_pointers(dA_displ,    dA_array, ldda, ZTRTRI_BATCHED_NB, 0, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array, lddb, ZTRTRI_BATCHED_NB, 0, batchCount, queue);                    
                    magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m-ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=ZTRTRI_BATCHED_NB; i < m; i += ZTRTRI_BATCHED_NB ) {
                        jb = min(m-i, ZTRTRI_BATCHED_NB);
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array, lddb,    i, 0, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array, lddx,    i, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, c_one, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+ZTRTRI_BATCHED_NB >= m)
                            break;
                        
                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda,  i+ZTRTRI_BATCHED_NB, i, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb,  i+ZTRTRI_BATCHED_NB, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m-i-ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // left, upper no-transpose
                // handle first block seperately with alpha
                jb = (m % ZTRTRI_BATCHED_NB == 0) ? ZTRTRI_BATCHED_NB : (m % ZTRTRI_BATCHED_NB);
                i = m-jb;
                magma_zdisplace_pointers(dinvA_displ,    dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dB_displ,          dB_array,    lddb, i, 0, batchCount, queue);
                magma_zdisplace_pointers(dX_displ,          dX_array,    lddx, i, 0, batchCount, queue);
                magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-ZTRTRI_BATCHED_NB >= 0) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=m-jb-ZTRTRI_BATCHED_NB; i >= 0; i -= ZTRTRI_BATCHED_NB ) {
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_one, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-ZTRTRI_BATCHED_NB < 0)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, i, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
        else {  // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // left, lower transpose
                // handle first block seperately with alpha
                jb = (m % ZTRTRI_BATCHED_NB == 0) ? ZTRTRI_BATCHED_NB : (m % ZTRTRI_BATCHED_NB);
                i = m-jb;
                magma_zdisplace_pointers(dinvA_displ,    dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dB_displ,          dB_array,    lddb, i, 0, batchCount, queue);
                magma_zdisplace_pointers(dX_displ,          dX_array,    lddx, i, 0, batchCount, queue);
                magma_zgemm_batched( transA, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-ZTRTRI_BATCHED_NB >= 0) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magma_zgemm_batched( transA, MagmaNoTrans, i, n, jb, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=m-jb-ZTRTRI_BATCHED_NB; i >= 0; i -= ZTRTRI_BATCHED_NB ) {
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magma_zgemm_batched( transA, MagmaNoTrans, ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_one, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-ZTRTRI_BATCHED_NB < 0)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0,  batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magma_zgemm_batched( transA, MagmaNoTrans, i, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // left, upper transpose
                // handle first block seperately with alpha
                jb = min(ZTRTRI_BATCHED_NB, m);
                magma_zgemm_batched( transA, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (ZTRTRI_BATCHED_NB < m) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda,      0,   ZTRTRI_BATCHED_NB, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, ZTRTRI_BATCHED_NB,        0, batchCount, queue);
                    magma_zgemm_batched( transA, MagmaNoTrans, m-ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=ZTRTRI_BATCHED_NB; i < m; i += ZTRTRI_BATCHED_NB ) {
                        jb = min(m-i, ZTRTRI_BATCHED_NB);
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magma_zgemm_batched( transA, MagmaNoTrans, jb, n, jb, c_one, dinvA_displ, ZTRTRI_BATCHED_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+ZTRTRI_BATCHED_NB >= m)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda,  i,        i+ZTRTRI_BATCHED_NB, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb,  i+ZTRTRI_BATCHED_NB,        0, batchCount, queue);
                        magma_zgemm_batched( transA, MagmaNoTrans, m-i-ZTRTRI_BATCHED_NB, n, ZTRTRI_BATCHED_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
    }
    else {  // side == MagmaRight
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag_batched( uplo, diag, n, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue);

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // right, lower no-transpose
                // handle first block seperately with alpha
                jb = (n % ZTRTRI_BATCHED_NB == 0) ? ZTRTRI_BATCHED_NB : (n % ZTRTRI_BATCHED_NB);
                i = n-jb;                
                magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-ZTRTRI_BATCHED_NB >= 0) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue);                        
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, i, jb, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=n-jb-ZTRTRI_BATCHED_NB; i >= 0; i -= ZTRTRI_BATCHED_NB ) {
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_one, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-ZTRTRI_BATCHED_NB < 0)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, i, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // right, upper no-transpose
                // handle first block seperately with alpha
                jb = min(ZTRTRI_BATCHED_NB, n);
                magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                if (ZTRTRI_BATCHED_NB < n) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, 0, ZTRTRI_BATCHED_NB, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, ZTRTRI_BATCHED_NB, batchCount, queue);
                    magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, n-ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=ZTRTRI_BATCHED_NB; i < n; i += ZTRTRI_BATCHED_NB ) {
                        jb = min(ZTRTRI_BATCHED_NB, n-i);
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, c_one, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+ZTRTRI_BATCHED_NB >= n)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, i, i+ZTRTRI_BATCHED_NB, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, i+ZTRTRI_BATCHED_NB, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m, n-i-ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
        else { // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // right, lower transpose
                // handle first block seperately with alpha
                jb = min(ZTRTRI_BATCHED_NB, n);
                magma_zgemm_batched( MagmaNoTrans, transA, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                if (ZTRTRI_BATCHED_NB < n) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda,  ZTRTRI_BATCHED_NB,      0, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb,       0, ZTRTRI_BATCHED_NB, batchCount, queue);
                    magma_zgemm_batched( MagmaNoTrans, transA, m, n-ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=ZTRTRI_BATCHED_NB; i < n; i += ZTRTRI_BATCHED_NB ) {
                        jb = min(ZTRTRI_BATCHED_NB, n-i);
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, transA, m, jb, jb, c_one, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+ZTRTRI_BATCHED_NB >= n)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda,  ZTRTRI_BATCHED_NB+i,        i, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb,         0, i+ZTRTRI_BATCHED_NB, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, transA, m, n-i-ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // right, upper transpose
                // handle first block seperately with alpha
                jb = (n % ZTRTRI_BATCHED_NB == 0) ? ZTRTRI_BATCHED_NB : (n % ZTRTRI_BATCHED_NB);
                i = n-jb;
                magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                magma_zgemm_batched( MagmaNoTrans, transA, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-ZTRTRI_BATCHED_NB >= 0) {
                    magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                    magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magma_zgemm_batched( MagmaNoTrans, transA, m, i, jb, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=n-jb-ZTRTRI_BATCHED_NB; i >= 0; i -= ZTRTRI_BATCHED_NB ) {
                        magma_zdisplace_pointers(dinvA_displ, dinvA_array, ZTRTRI_BATCHED_NB, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_zdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, transA, m, ZTRTRI_BATCHED_NB, ZTRTRI_BATCHED_NB, c_one, dB_displ, lddb, dinvA_displ, ZTRTRI_BATCHED_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-ZTRTRI_BATCHED_NB < 0)
                            break;

                        magma_zdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                        magma_zdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magma_zgemm_batched( MagmaNoTrans, transA, m, i, ZTRTRI_BATCHED_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
    }
}
/***************************************************************************//**
    Purpose
    -------
    ztrsm_work solves one of the matrix equations on gpu

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
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

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
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of dimension ( ldda, k ), where k is m
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
            On entry, ldda specifies the first dimension of each array A.
            When side = MagmaLeft,  ldda >= max( 1, m ),
            when side = MagmaRight, ldda >= max( 1, n ).

    @param[in,out]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of dimension ( lddb, n ).
             Before entry, the leading m by n part of the array B must
             contain the right-hand side matrix B.
             \n
             On exit, the solution matrix X

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B.
            lddb >= max( 1, m ).

    @param[in,out]
    dX_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array X of dimension ( lddx, n ).
             On entry, should be set to 0
             On exit, the solution matrix X

    @param[in]
    lddx    INTEGER.
            On entry, lddx specifies the first dimension of each array X.
            lddx >= max( 1, m ).

    @param
    dinvA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array dinvA, a workspace on device.
            If side == MagmaLeft,  dinvA must be of size >= ceil(m/ZTRTRI_BATCHED_NB)*ZTRTRI_BATCHED_NB*ZTRTRI_BATCHED_NB,
            If side == MagmaRight, dinvA must be of size >= ceil(n/ZTRTRI_BATCHED_NB)*ZTRTRI_BATCHED_NB*ZTRTRI_BATCHED_NB,
            where ZTRTRI_BATCHED_NB = 128.

    @param[in]
    dinvA_length    INTEGER
                   The size of each workspace matrix dinvA
                   
    @param
    dA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dB_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dX_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dinvA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param[in]
    resetozero INTEGER
               Used internally by ZTRTRI_DIAG routine
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C"
void magmablas_ztrsm_inv_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx, 
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, 
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, magma_queue_t queue)
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
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    magmablas_ztrsm_inv_outofplace_batched( 
                    side, uplo, transA, diag, flag,
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, dinvA_length,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, queue );
    // copy X to B
    magma_zdisplace_pointers(dX_displ,    dX_array, lddx, 0, 0, batchCount, queue);
    magma_zdisplace_pointers(dB_displ,    dB_array, lddb, 0, 0, batchCount, queue);
    magmablas_zlacpy_batched( MagmaFull, m, n, dX_displ, lddx, dB_displ, lddb, batchCount, queue );
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
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of dimension ( ldda, k ), where k is m
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
            On entry, ldda specifies the first dimension of each array A.
            When side = MagmaLeft,  ldda >= max( 1, m ),
            when side = MagmaRight, ldda >= max( 1, n ).

    @param[in,out]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of dimension ( lddb, n ).
             Before entry, the leading m by n part of the array B must
             contain the right-hand side matrix B.
             \n
             On exit, the solution matrix X

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B.
            lddb >= max( 1, m ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C"
void magmablas_ztrsm_inv_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue)
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
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    magmaDoubleComplex **dA_displ     = NULL;
    magmaDoubleComplex **dB_displ     = NULL;
    magmaDoubleComplex **dX_displ     = NULL;
    magmaDoubleComplex **dinvA_displ  = NULL;
    magmaDoubleComplex **dX_array     = NULL;
    magmaDoubleComplex **dinvA_array  = NULL;

    magma_malloc((void**)&dA_displ,    batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dB_displ,    batchCount * sizeof(*dB_displ));
    magma_malloc((void**)&dX_displ,    batchCount * sizeof(*dX_displ));
    magma_malloc((void**)&dinvA_displ, batchCount * sizeof(*dinvA_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dX_array,    batchCount * sizeof(*dX_array));

    magma_int_t size_dinvA;
    magma_int_t lddx = m;
    magma_int_t size_x = lddx*n;

    if ( side == MagmaLeft ) {
        size_dinvA = magma_roundup( m, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    }
    else {
        size_dinvA = magma_roundup( n, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    }
    magmaDoubleComplex *dinvA=NULL, *dX=NULL;
    magma_int_t resetozero = 0;
    magma_zmalloc( &dinvA, size_dinvA*batchCount );
    magma_zmalloc( &dX, size_x*batchCount );
    if ( dinvA == NULL || dX == NULL ) {
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return;
    }
    magmablas_zlaset(MagmaFull, size_dinvA, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA, size_dinvA, queue);
    magmablas_zlaset(MagmaFull, lddx, n*batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dX, lddx, queue);

    magma_zset_pointer( dX_array, dX, lddx, 0, 0, size_x, batchCount, queue );
    magma_zset_pointer( dinvA_array, dinvA, ZTRTRI_BATCHED_NB, 0, 0, size_dinvA, batchCount, queue );

    magmablas_ztrsm_inv_work_batched( 
                    side, uplo, transA, diag, 1, 
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, size_dinvA,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, queue );

    magma_free( dinvA );
    magma_free( dX );
    magma_free(dA_displ);
    magma_free(dB_displ);
    magma_free(dX_displ);
    magma_free(dinvA_displ);
    magma_free(dinvA_array);
    magma_free(dX_array);
}
