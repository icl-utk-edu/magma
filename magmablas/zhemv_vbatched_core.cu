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
#include "hemv_template_kernel_vbatched.cuh"

/******************************************************************************/
extern "C" void 
magmablas_zhemv_vbatched_core(
        magma_uplo_t uplo, magma_int_t* n, 
        magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda,
                                  magmaDoubleComplex **dX_array, magma_int_t* incx,
        magmaDoubleComplex beta,  magmaDoubleComplex **dY_array, magma_int_t* incy,
        magma_int_t max_n, 
        magma_int_t offA, magma_int_t offX, magma_int_t offY, 
        magma_int_t spec_n, 
        magma_int_t batchCount, magma_queue_t queue )
{
    if(uplo == MagmaLower){
        const int param[] = {ZHEMV_BATCHED_LOWER};
        const int nb = param[0];
        hemv_diag_template_vbatched<magmaDoubleComplex, ZHEMV_BATCHED_LOWER>
                ( uplo, n, 
                  alpha, dA_array, ldda, 
                         dX_array, incx, 
                  beta,  dY_array, incy, 
                  max_n, offA, offX, offY, spec_n, batchCount, queue);
        if(max_n > nb){
            hemv_lower_template_vbatched<magmaDoubleComplex, ZHEMV_BATCHED_LOWER>
                ( n, alpha, 
                  dA_array, ldda, 
                  dX_array, incx, 
                  dY_array, incy, 
                  max_n, offA, offX, offY, spec_n, batchCount, queue);
        }
    }
    else{    // upper
        const int param[] = {ZHEMV_BATCHED_UPPER};
        const int nb = param[0];
        hemv_diag_template_vbatched<magmaDoubleComplex, ZHEMV_BATCHED_UPPER>
                ( uplo, n, 
                  alpha, dA_array, ldda, 
                         dX_array, incx, 
                  beta,  dY_array, incy, 
                  max_n, offA, offX, offY, spec_n, batchCount, queue);
        if(max_n > nb){
            hemv_upper_template_vbatched<magmaDoubleComplex, ZHEMV_BATCHED_UPPER>
                ( n, alpha, 
                  dA_array, ldda, 
                  dX_array, incx, 
                  dY_array, incy, 
                  max_n, offA, offX, offY, spec_n, batchCount, queue);
        }
    }
}
/******************************************************************************/
extern "C" void 
magmablas_zhemv_vbatched_max_nocheck(
        magma_uplo_t uplo, magma_int_t* n, 
        magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda,
                                  magmaDoubleComplex **dX_array, magma_int_t* incx,
        magmaDoubleComplex beta,  magmaDoubleComplex **dY_array, magma_int_t* incy, 
        magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue )
{
    magmablas_zhemv_vbatched_core( 
            uplo, n, 
            alpha, dA_array, ldda, 
                   dX_array, incx,
            beta,  dY_array, incy,  
            max_n, 0, 0, 0, 0, 
            batchCount, queue );
}
/******************************************************************************/
