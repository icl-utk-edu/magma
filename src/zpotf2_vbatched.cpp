/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_z
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magmaDoubleComplex **dA_array, magma_int_t* lda,
    magmaDoubleComplex **dA_displ, 
    magmaDoubleComplex **dW_displ,
    magmaDoubleComplex **dB_displ, 
    magmaDoubleComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo=0;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else{
        arginfo = magma_zpotrf_lpout_vbatched(uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue);
    }

    return arginfo;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
