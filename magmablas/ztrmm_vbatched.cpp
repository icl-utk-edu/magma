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
magmablas_ztrmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue )
{
    if ( max_m <= 0 || max_n <= 0 )
        return;
    
    magmablas_ztrmm_vbatched_core( 
            side, uplo, transA, diag, 
            m, n, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            max_m, max_n,  
            0, 0, 0, 0, 
            0, 0, 
            batchCount, queue );
}

/******************************************************************************/
extern "C" void
magmablas_ztrmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue )
{
    magma_int_t info = 0;
    info =  magma_trmm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue);
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_ztrmm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, 
            alpha, dA_array, ldda,
                   dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue );
}

/******************************************************************************/
extern "C" void
magmablas_ztrmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue )
{
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_ztrmm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, 
            alpha, dA_array, ldda,
                   dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue );
}

/***************************************************************************//**
    Purpose   
    =======   

    ZTRMM  performs one of the matrix-matrix operations   

       B := alpha*op( A )*B,   or   B := alpha*B*op( A )   

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or   
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of 

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).   

    Parameters   
    ==========   

    @param[in]
    side     magma_side_t.
             On entry,  side specifies whether  op( A ) multiplies B from 
             the left or right as follows:   
                side = magmaLeft   B := alpha*op( A )*B.   
                side = magmaRight  B := alpha*B*op( A ).   
             Unchanged on exit.   

    @param[in]
    uplo     magma_uplo_t.
             On entry, uplo specifies whether the matrix A is an upper or 
             lower triangular matrix as follows:   
                uplo = magmaUpper   A is an upper triangular matrix.   
                uplo = magmaLower   A is a lower triangular matrix.   
             Unchanged on exit.   

    @param[in]
    transA   magma_trans_t.
             On entry, transA specifies the form of op( A ) to be used in 
             the matrix multiplication as follows:   
                transA = MagmaNoTrans     op( A ) = A.   
                transA = MagmaTrans       op( A ) = A'.   
                transA = MagmaConjTrans   op( A ) = conjg( A' ).   
             Unchanged on exit.   

    @param[in]
    diag     magma_diag_t.
             On entry, diag specifies whether or not A is unit triangular 
             as follows:   
                diag = MagmaUnit      A is assumed to be unit triangular.   
                diag = MagmaNonUnit   A is not assumed to be unit   
                                    triangular.   
             Unchanged on exit.   

    @param[in]
    m        INTEGER array, dimension(batchCount + 1).
             On entry, each integer M specifies the number of rows of the corresponding 
             matrix B. M must be at least zero.   
             Unchanged on exit.   

    @param[in]
    n        INTEGER array, dimension(batchCount + 1).   
             On entry, each integer n specifies the number of columns of the corresponding 
             matrix B.  N must be at least zero.   
             Unchanged on exit.   

    @param[in]
    alpha    DOUBLE COMPLEX.
             On entry,  alpha specifies the scalar  alpha. When  alpha is 
             zero then  A is not referenced and  B need not be set before 
             entry.   
             Unchanged on exit.   

    @param[in]
    dA_array     Array of pointers, dimension(batchCount).
             Each is a DOUBLE COMPLEX array A of DIMENSION ( ldda, k ), where k is M 
             when  side = magmaLeft  and is  N  when  side = magmaRight. 
             Before entry  with  uplo = magmaUpper,  the  leading  k by k 
             upper triangular part of the array  A must contain the upper 
             triangular matrix  and the strictly lower triangular part of 
             A is not referenced.   
             Before entry  with  uplo = magmaLower,  the  leading  k by k 
             lower triangular part of the array  A must contain the lower 
             triangular matrix  and the strictly upper triangular part of 
             A is not referenced.   
             Note that when  diag = MagmaUnit,  the diagonal elements of 
             A  are not referenced either,  but are assumed to be  unity. 
             Unchanged on exit.   

    @param[in]
    ldda     INTEGER array, dimension(batchCount + 1).
             On entry, ldda specifies the first dimension of A as declared 
             in the calling (sub) program.  When  side = magmaLeft  then 
             ldda  must be at least  max( 1, M ),  when  side = magmaRight 
             then ldda must be at least max( 1, N ).   
             Unchanged on exit.   

    @param[in,out]
    dB_array     Array of pointers, dimension(batchCount).
             Each is a DOUBLE COMPLEX array B of DIMENSION ( lddb, N ).   
             Before entry,  the leading  M by N part of the array  B must 
             contain the matrix  B,  and  on exit  is overwritten  by the 
             transformed matrix.   

    @param[in]
    lddb     INTEGER array, dimension(batchCount + 1).
             On entry, lddb specifies the first dimension of B as declared 
             in  the  calling  (sub)  program.   lddb  must  be  at  least 
             max( 1, M ).   
             Unchanged on exit.   

    @param[in]
    batchCount  INTEGER.
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t.
            Queue to execute in.

    @ingroup magma_trmm_batched
*******************************************************************************/
extern "C" void
magmablas_ztrmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    info =  magma_trmm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue);
        
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

    magmablas_ztrmm_vbatched_max_nocheck(
            side, uplo, transA, diag, 
            m, n, 
            alpha, dA_array, ldda,
                   dB_array, lddb, 
            batchCount, 
            max_m, max_n, queue );
}
