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
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

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
    ZHEMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n Hermitian matrix.
    This is the variable size batched version of the operation. 

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is to be referenced as
            follows:
      -     = MagmaUpper:  Only the upper triangular part of A is to be referenced.
      -     = MagmaLower:  Only the lower triangular part of A is to be referenced.

    @param[in]
    n       INTEGER array, dimension(batchCoutn + 1).
            On entry, each element N specifies the order of each matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX_16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array A of DIMENSION ( LDDA, N ).
            Before entry with UPLO = MagmaUpper, the leading N by N
            upper triangular part of the array A must contain the upper
            triangular part of the Hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = MagmaLower, the leading N by N
            lower triangular part of the array A must contain the lower
            triangular part of the Hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.

    @param[in]
    ldda    INTEGER array, dimension(batchCount + 1).
            On entry, each element LDDA specifies the first dimension of each A as declared
            in the calling (sub) program. LDDA must be at least
            max( 1, n ).
            It is recommended that LDDA is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    dx_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array X of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector X.

    @param[in]
    incx    INTEGER array, dimension(batchCount + 1).
            On entry, each element INCX specifies the increment for the elements of
            each X. INCX must not be zero.

    @param[in]
    beta    COMPLEX_16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    dy_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array Y of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector Y. On exit, Y is overwritten by the updated
            vector Y.

    @param[in]
    incy    INTEGER array, dimension(batchCount + 1).
            On entry, each element INCY specifies the increment for the elements of
            each Y. INCY must not be zero.

    @param[in]
    batchCount    INTEGER.
            The number of problems to operate on. 
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_hemv_batched
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
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_zhemv_vbatched_max_nocheck( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_n, queue);
}
