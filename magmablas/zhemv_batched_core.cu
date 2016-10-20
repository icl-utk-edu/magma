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
#include "batched_kernel_param.h"

#define PRECISION_z
#include "hemv_template_kernel_batched.cuh"

/******************************************************************************/
extern "C" void 
magmablas_zhemv_batched_core(
        magma_uplo_t uplo, magma_int_t n, 
        magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda,
                                  magmaDoubleComplex **dX_array, magma_int_t incx,
        magmaDoubleComplex beta,  magmaDoubleComplex **dY_array, magma_int_t incy,
        magma_int_t offA, magma_int_t offX, magma_int_t offY, 
        magma_int_t batchCount, magma_queue_t queue )
{
    if(uplo == MagmaLower){
        const int param[] = {ZHEMV_BATCHED_LOWER};
        const int nb = param[0];
        hemv_diag_template_batched<magmaDoubleComplex, ZHEMV_BATCHED_LOWER>
                ( uplo, n, 
                  alpha, dA_array, ldda, 
                         dX_array, incx, 
                  beta,  dY_array, incy, 
                  offA, offX, offY, batchCount, queue);
        if(n > nb){
            hemv_lower_template_batched<magmaDoubleComplex, ZHEMV_BATCHED_LOWER>
                ( n, alpha, 
                  dA_array, ldda, 
                  dX_array, incx, 
                  dY_array, incy, 
                  offA, offX, offY, batchCount, queue);
        }
    }
    else{    // upper
        const int param[] = {ZHEMV_BATCHED_UPPER};
        const int nb = param[0];
        hemv_diag_template_batched<magmaDoubleComplex, ZHEMV_BATCHED_UPPER>
                ( uplo, n, 
                  alpha, dA_array, ldda, 
                         dX_array, incx, 
                  beta,  dY_array, incy, 
                  offA, offX, offY, batchCount, queue);
        if(n > nb){
            hemv_upper_template_batched<magmaDoubleComplex, ZHEMV_BATCHED_UPPER>
                ( n, alpha, 
                  dA_array, ldda, 
                  dX_array, incx, 
                  dY_array, incy, 
                  offA, offX, offY, batchCount, queue);
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZHEMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n Hermitian matrix.
    This is the fixed size batched version of the operation. 

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
    n       INTEGER.
            On entry, N specifies the order of each matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX_16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array A of DIMENSION ( LDDA, n ).
            Before entry with UPLO = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular part of the Hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular part of the Hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.

    @param[in]
    ldda    INTEGER.
            On entry, LDDA specifies the first dimension of each A as declared
            in the calling (sub) program. LDDA must be at least
            max( 1, n ).
            It is recommended that ldda is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    dX_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array X of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector X.

    @param[in]
    incx    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    beta    COMPLEX_16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    dY_array    Array of pointers, dimension(batchCount). 
            Each is a COMPLEX_16 array Y of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector Y. On exit, Y is overwritten by the updated
            vector Y.

    @param[in]
    incy    INTEGER.
            On entry, INCY specifies the increment for the elements of
            Y. INCY must not be zero.

    @param[in]
    batchCount    INTEGER.
            The number of problems to operate on. 
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_hemv_batched
*******************************************************************************/
extern "C" void 
magmablas_zhemv_batched(
        magma_uplo_t uplo, magma_int_t n, 
        magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda,
                                  magmaDoubleComplex **dX_array, magma_int_t incx,
        magmaDoubleComplex beta,  magmaDoubleComplex **dY_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper ) {
        info = -1;
    } else if ( n < 0 ) {
        info = -2;
    } else if ( ldda < max(1, n) ) {
        info = -5;
    } else if ( incx == 0 ) {
        info = -7;
    } else if ( incy == 0 ) {
        info = -10;
    } else if ( batchCount < 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return;    
    
    magmablas_zhemv_batched_core( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dX_array, incx,
            beta,  dY_array, incy,  
            0, 0, 0, 
            batchCount, queue );
}
///////////////////////////////////////////////////////////////////////////////////////////////////
