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
magmablas_zhemv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    info =  magma_hemv_vbatched_checker( uplo, n, ldda, incx, incy, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_zhemv_vbatched_max_nocheck( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_n, queue);
}


/******************************************************************************/
extern "C" void
magmablas_zhemv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    // compute the max. dimensions
    magma_imax_size_1(n, batchCount, queue);
    magma_int_t max_n; 
    magma_getvector(1, sizeof(magma_int_t), &n[batchCount], 1, &max_n, 1, queue);
    
    magmablas_zhemv_vbatched_max_nocheck( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_n, queue);
}


/***************************************************************************//**
    Purpose
    -------
    ZHEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER M specifies the number of rows of each matrix A.
            The last element of the array is used internally by the routine. 

    @param[in]
    n       Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER N specifies the number of columns of each matrix A
            The last element of the array is used internally by the routine. 
 
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.


    @param[in]
    dA_array 	Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of DIMENSION ( LDDA, N ) on the GPU
   
    @param[in]
    ldda    Array of integers, dimension (batchCount + 1).
            Each INTEGER LDDA specifies the leading dimension of each matrix A.

    @param[in]
    dx_array 	Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension
            N if trans == MagmaNoTrans
            M if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Array of integers, dimension (batchCount + 1).
            Each integer specifies the increment for the elements of each vector X.
            INCX must not be zero.
            The last element of the array is used internally by the routine. 
  
    @param[in]
    beta    COMPLEX_16
            On entry, ALPHA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy_array 	Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension
            M if trans == MagmaNoTrans
            N if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Array of integers, dimension (batchCount + 1).
            Each integer specifies the increment for the elements of each vector Y.
            INCY must not be zero.
            The last element of the array is used internally by the routine. 

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv_batched
*******************************************************************************/
extern "C" void
magmablas_zhemv_vbatched(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    info =  magma_hemv_vbatched_checker( uplo, n, ldda, incx, incy, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // compute the max. dimensions
    magma_imax_size_1(n, batchCount, queue);
    magma_int_t max_n; 
    magma_getvector(1, sizeof(magma_int_t), &n[batchCount], 1, &max_n, 1, queue);
    
    magmablas_zhemv_vbatched_max_nocheck( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_n, queue);
}
