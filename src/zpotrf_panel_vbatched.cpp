/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#define PRECISION_z

#include "batched_kernel_param.h"
////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magma_int_t *ibvec, magma_int_t nb,  
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dX_array,    magma_int_t* dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ, 
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t *n_minus_ib;
    magma_imalloc( &n_minus_ib, batchCount );
    arginfo = magma_zpotf2_vbatched(
                       uplo, ibvec, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, queue);

    if ((max_n-nb) > 0) {
        // n-ib
        magma_ivec_add( batchCount, 1, n, -1, ibvec, n_minus_ib, queue);
        magma_zdisplace_pointers_var_cc(dW0_displ, dA_array, ldda, nb, 0, batchCount, queue);
        magmablas_ztrsm_work_vbatched( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 1,  
                                      n_minus_ib, ibvec, 
                                      MAGMA_Z_ONE, 
                                      dA_array,  ldda, 
                                      dW0_displ, ldda, 
                                      dX_array,  n_minus_ib, 
                                      dinvA_array, dinvA_length, 
                                      dW1_displ, dW2_displ, 
                                      dW3_displ, dW4_displ, 
                                      0, batchCount, 
                                      max_n-nb, nb, queue );
    }
    magma_free( n_minus_ib );
    return arginfo;
}
////////////////////////////////////////////////////////////////////////////////////////
