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

///////////////////////////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
    TODO: Documentation 
*/
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
