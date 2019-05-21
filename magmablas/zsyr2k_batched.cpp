/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#include "magma_internal.h"
#include "commonblas_z.h"

#define PRECISION_z

/***************************************************************************//**
    Purpose
    -------
    ZSYR2K  performs one of the symmetric rank 2k operations
    
        C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
    
    or
    
        C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
    
    where  alpha and beta  are scalars with  beta  real,  C is an  n by n
    symmetric matrix and  A and B  are  n by k matrices in the first case
    and  k by n  matrices in the second case.
    
    Parameters
    ----------
    @param[in]
    uplo     magma_uplo_t.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array C is to be referenced as
             follows:
      -     = MagmaUpper:  Only the upper triangular part of C is to be referenced.
      -     = MagmaLower:  Only the lower triangular part of C is to be referenced.

    @param[in]
    trans    magma_trans_t.
             On entry, TRANS specifies the operation to be performed as
             follows:
      -     = MagmaNoTrans:     C := alpha*A*B**H + conj( alpha )*B*A**H + beta*C.
      -     = Magma_ConjTrans:  C := alpha*A**H*B + conj( alpha )*B**H*A + beta*C.

    @param[in]
    n        INTEGER.
             On entry, N specifies the order of the matrix C. N must be
             at least zero.

    @param[in]
    k        INTEGER.
             On entry with TRANS = MagmaNoTrans, k specifies the number
             of columns of the matrices A and B, and on entry with
             TRANS = Magma_ConjTrans, k specifies the number of rows of the
             matrices A and B. k must be at least zero.

    @param[in]
    alpha    COMPLEX*16.
             On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array of DIMENSION ( ldda, ka ), where ka is
             k when TRANS = MagmaNoTrans, and is n otherwise.
             Before entry with TRANS = MagmaNoTrans, the leading n by k
             part of the array A must contain the matrix A, otherwise
             the leading k by n part of the array A must contain the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
             in the calling (sub) program. When TRANS = MagmaNoTrans
             then ldda must be at least max( 1, n ), otherwise ldda must
             be at least max( 1, k ).
    
    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array of DIMENSION ( ldb, kb ), where kb is
             k  when  TRANS = MagmaNoTrans,  and is  n  otherwise.
             Before entry with  TRANS = MagmaNoTrans,  the  leading  n by k
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  k by n  part of the array  B  must contain  the
             matrix B.
    
    @param[in]
    lddb     INTEGER
             On entry, lddb specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   When  TRANS = MagmaNoTrans
             then  lddb must be at least  max( 1, n ), otherwise  lddb must
             be at least  max( 1, k ).
             Unchanged on exit.
    
    
    @param[in]
    beta    COMPLEX*16.
            On entry,  BETA  specifies the scalar  beta.  
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is COMPLEX*16 array of DIMENSION ( lddc, n ).
             Before entry  with  UPLO = MagmaUpper,  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = MagmaLower,  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.
             Note that the imaginary parts of the diagonal elements need
             not be set,  they are assumed to be zero,  and on exit they
             are set to zero.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, n ).
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_syr2k_batched
*******************************************************************************/
extern "C" void
magmablas_zsyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb, 
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex c_one = MAGMA_Z_MAKE(1., 0.);
    
    if ( uplo != MagmaLower && uplo != MagmaUpper) {
        info = -1; 
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans) {
        info = -2;
    } else if ( n < 0 ) {
        info = -3;
    } else if ( k < 0 ) {
        info = -4;
    } else if ( ((trans == MagmaNoTrans) && ldda < max(1,n)) ||
                ((trans != MagmaNoTrans) && ldda < max(1,k)) ) {
        info = -7;
    } else if ( ((trans == MagmaNoTrans) && lddb < max(1,n)) ||
                ((trans != MagmaNoTrans) && lddb < max(1,k)) ) {
        info = -9;
    } else if ( lddc < max(1,n) ) {
        info = -12;
    } else if ( batchCount < 0 ) {
        info = -13;
    }
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // Quick return if possible
    if( ( n == 0 ) || 
        ( (alpha == 0 || k == 0) && (beta == 1) ) || 
        ( batchCount == 0 ))
    {
        return;
    }
    
    if( trans == MagmaNoTrans){
        magmablas_zsyrk_batched_core(
                uplo, MagmaNoTrans, 
                n, k, 
                alpha, dA_array, 0, 0, ldda, 
                       dB_array, 0, 0, lddb, 
                beta,  dC_array, 0, 0, lddc, 
                batchCount, queue );
        magmablas_zsyrk_batched_core(
                uplo, MagmaNoTrans, 
                n, k, 
                alpha, dB_array, 0, 0, lddb, 
                       dA_array, 0, 0, ldda, 
                c_one, dC_array, 0, 0, lddc, 
                batchCount, queue );    
    }else{
        magmablas_zsyrk_batched_core(
                uplo, MagmaConjTrans, 
                n, k, 
                alpha, dA_array, 0, 0, ldda, 
                       dB_array, 0, 0, lddb, 
                beta,  dC_array, 0, 0, lddc, 
                batchCount, queue );
        magmablas_zsyrk_batched_core(
                uplo, MagmaConjTrans, 
                n, k, 
                alpha, dB_array, 0, 0, lddb, 
                       dA_array, 0, 0, ldda, 
                c_one, dC_array, 0, 0, lddc, 
                batchCount, queue );
    }
}
