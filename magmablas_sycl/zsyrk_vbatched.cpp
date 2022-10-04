/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c
       
       @author Ahmad Abdelfattah
*/
#include "magma_internal.h"
#include "commonblas_z.h"

/******************************************************************************/
extern "C" void
magmablas_zsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info =  magma_syrk_vbatched_checker( 1, uplo, trans, n, k, ldda, lddc, batchCount, queue );
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_zsyrk_vbatched_max_nocheck( 
            uplo, trans, 
            n, k, 
            alpha, dA_array, ldda, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_n, max_k, queue );
}


/******************************************************************************/
extern "C" void
magmablas_zsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue )
{
    // compute the max. dimensions
    magma_imax_size_2(n, k, batchCount, queue);
    magma_int_t max_n, max_k; 
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_igetvector_async(1, &k[batchCount], 1, &max_k, 1, queue);
    magma_queue_sync( queue );

    magmablas_zsyrk_vbatched_max_nocheck( 
            uplo, trans, 
            n, k, 
            alpha, dA_array, ldda, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_n, max_k, queue );
}


/***************************************************************************//**
    Purpose
    -------
    ZSYRK  performs one of the symmetric rank k operations
    
        C := alpha*A*A**T + beta*C,
    or
        
        C := alpha*A**T*A + beta*C,
    
    where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix
    in the second case.
    
    Parameters
    ----------
    @param[in]
    uplo    magma_uplo_t.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = MagmaUpper Only the upper triangular part of C
           is to be referenced.

           uplo = MagmaLower Only the lower triangular part of C
           is to be referenced.
    
    @param[in]
    trans   magma_trans_t.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = MagmaNoTrans C := alpha*A*A**T + beta*C.

            trans = MagmaConjTrans C := alpha*A**T*A + beta*C.

    @param[in]
    n       Array of integers, size (batchCount + 1).
            On entry,  each INTEGER N specifies the order of the corresponding matrix C. 
            N must be at least zero. 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    k       Array of integers, size (batchCount + 1).
            On entry with trans = MagmaNoTrans, each INTEGER K specifies the number
            of columns of the corresponding matrix A, and on entry with
            trans = MagmaConjTrans, K specifies the number of rows of the
            corresponding matrix A. K must be at least zero. 
            The last element of the array is used internally by the routine. 

    @param[in]
    alpha   COMPLEX_16.
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array     Array of pointers, size (batchCount).
            Each is a COMPLEX_16 array of DIMENSION ( LDDA, Ka ), where Ka is
            K  when  trans = MagmaNoTrans,  and is  N  otherwise.
            Before entry with  trans = MagmaNoTrans,  the leading  N by K
            part of the corresponding array A must contain the matrix A, otherwise
            the leading  K by N  part of the corresponding array must contain the
            matrix A.
    
    @param[in]
    ldda    Array of integers, size (batchCount + 1).
            On entry, each INTEGER LDDA specifies the first dimension of the corresponding 
            matrix A as declared in the calling (sub) program. When trans = MagmaNoTrans then
            LDDA must be at least  max( 1, N ), otherwise  ldda must be at
            least  max( 1, K ). 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    beta    COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.
    
    @param[in,out]
    dC_array     Array of pointers, size (batchCount).
            Each is a COMPLEX_16 array of DIMENSION ( LDDC, N ).
            Before entry with uplo = MagmaUpper, the leading N by N
            upper triangular part of the corresponding array C must 
            contain the upper triangular part of the corresponding 
            symmetric matrix and the strictly lower triangular part of C 
            is not referenced. On exit, the upper triangular part of the 
            array C is overwritten by the upper triangular part of 
            the updated matrix.
            Before entry with uplo = MagmaLower, the leading N by N
            lower triangular part of the corresponding array C must 
            contain the lower triangular part of the corresponding 
            symmetric matrix and the strictly upper triangular part 
            of C is not referenced. On exit, the lower triangular 
            part of the array C is overwritten by the lower triangular 
            part of the updated matrix.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero, and on exit they
            are set to zero.

    @param[in]
    lddc    Array of integers, size (batchCount + 1).
            On entry, each INTEGER LDDC specifies the first dimension of the 
            corresponding matrix C as declared in  the  calling  (sub)  program. 
            LDDC  must  be  at  least max( 1, M ).
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_syrk_batched
*******************************************************************************/
extern "C" void
magmablas_zsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info =  magma_syrk_vbatched_checker( 1, uplo, trans, n, k, ldda, lddc, batchCount, queue );
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // compute the max. dimensions
    magma_imax_size_2(n, k, batchCount, queue);
    magma_int_t max_n, max_k; 
    magma_getvector(1, sizeof(magma_int_t), &n[batchCount], 1, &max_n, 1, queue);
    magma_getvector(1, sizeof(magma_int_t), &k[batchCount], 1, &max_k, 1, queue);
    
    magmablas_zsyrk_vbatched_max_nocheck( 
            uplo, trans, 
            n, k, 
            alpha, dA_array, ldda, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_n, max_k, queue );
}
