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
magmablas_zgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_queue_t queue )
{
    magmablas_zgemm_vbatched_core(
            transA, transB, 
            m, n, k, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            max_m, max_n, max_k, 
            0, 0, 0, 0, 0, 0, // row/col offsets for A, B, and C
            0, 0, 0,          // specific m, n, k values (ignored if <= 0)
            batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_zgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info =  magma_gemm_vbatched_checker( transA, transB, m, n, k, ldda, lddb, lddc, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_zgemm_vbatched_max_nocheck(
            transA, transB, 
            m, n, k, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_m, max_n, max_k, 
            queue );
}


/******************************************************************************/
extern "C" void
magmablas_zgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    // compute the max. dimensions
    magma_imax_size_3(m, n, k, batchCount, queue);
    magma_int_t max_m, max_n, max_k; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_igetvector_async(1, &k[batchCount], 1, &max_k, 1, queue);
    magma_queue_sync( queue );

    magmablas_zgemm_vbatched_max_nocheck(
            transA, transB, 
            m, n, k, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_m, max_n, max_k, 
            queue );
}


/***************************************************************************//**
    Purpose
    -------
    ZGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.
    
    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.
    
    @param[in]
    m       Array of integers, dimension (batchCount + 1)
            On entry,  each INTEGER M  specifies  the number  of rows  of the 
            corresponding matrices op( A ) and C.  M  must  be at least  zero. 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    n       Array of integers, dimension (batchCount + 1).
            On entry,  each INTEGER N  specifies the number  of columns of the 
            corresponding matrix op( B ) and the number of columns of the 
            corresponding matrix C. N must be at least zero. 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    k       Array of integers, dimension (batchCount + 1).
            On entry,  each INTEGER K  specifies  the number of columns of the 
            corresponding matrix op( A ) and the number of rows of the 
            corresponding matrix op( B ). K must be at least  zero.
            The last element of the array is used internally by the routine. 
    
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of DIMENSION ( LDDA, ka ), where ka is
             K  when  transA = MagmaNoTrans,  and is  M  otherwise.
             Before entry with  transA = MagmaNoTrans,  the leading  M by K
             part of the array A must contain the matrix A, otherwise
             the leading  K by M  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER LDDA specifies the first dimension of 
            the corresponding matrix A as declared in the calling (sub) program. 
            When  transA = MagmaNoTrans then LDDA must be at least  max( 1, M ), 
            otherwise  ldda must be at least  max( 1, K ). 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of DIMENSION ( LDDB, kb ), where kb is
             N  when  transB = MagmaNoTrans,  and is  K  otherwise.
             Before entry with  transB = MagmaNoTrans,  the leading  K by N
             part of the array B must contain the matrix B, otherwise
             the leading  N by K  part of the array B must contain  the
             matrix B.
    
    @param[in]
    lddb    Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER LDDB specifies the first dimension of 
            the corresponding matrix B as declared in the calling (sub) program. 
            When  transB = MagmaNoTrans then LDDB must be at least  max( 1, K ), 
            otherwise  LDDB must be at least  max( 1, N ).
            The last element of the array is used internally by the routine. 
    
    @param[in]
    beta    COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array C of DIMENSION ( LDDC, N ).
             Before entry, the leading  M by N  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  M by N  matrix
             ( alpha*op( A )*op( B ) + beta*C ).
    
    @param[in]
    lddc    Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER LDDC specifies the first dimension of 
            the corresponding matrix C as declared in  the  calling  (sub)  program. 
            LDDC  must  be  at  least max( 1, M ). 
            The last element of the array is used internally by the routine. 
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_gemm_batched
*******************************************************************************/
extern "C" void
magmablas_zgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    info =  magma_gemm_vbatched_checker( transA, transB, m, n, k, ldda, lddb, lddc, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // compute the max. dimensions
    magma_imax_size_3(m, n, k, batchCount, queue);
    magma_int_t max_m, max_n, max_k; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_igetvector_async(1, &k[batchCount], 1, &max_k, 1, queue);
    magma_queue_sync( queue );
    
    magmablas_zgemm_vbatched_max_nocheck(
            transA, transB, 
            m, n, k, 
            alpha, dA_array, ldda, 
                   dB_array, lddb, 
            beta,  dC_array, lddc, 
            batchCount, 
            max_m, max_n, max_k, 
            queue );
}
