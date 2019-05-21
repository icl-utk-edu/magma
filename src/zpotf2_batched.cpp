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

#define COMPLEX

/******************************************************************************/
// This is a recursive routine
extern "C" magma_int_t
magma_zpotf2_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo=0;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;

    magma_int_t crossover = magma_get_zpotrf_batched_crossover();

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
    }
    else {
        if( n <= crossover ){
            arginfo = magma_zpotrf_lpout_batched(uplo, n, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);
        }
        else{
            magma_int_t n1 = n / 2;
            magma_int_t n2 = n - n1;
            // panel
            magma_zpotrf_lpout_batched(uplo, n1, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);

            // trsm
            magmablas_ztrsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n2, n1, MAGMA_Z_ONE, 
                    dAarray(ai   , aj), ldda, 
                    dAarray(ai+n1, aj), ldda, batchCount, queue );

            // herk
            magmablas_zherk_batched_core( 
                    MagmaLower, MagmaNoTrans, 
                    n2, n1, 
                    c_neg_one, dAarray(ai+n1, aj   ), ldda,
                               dAarray(ai+n1, aj   ), ldda,
                    c_one,     dAarray(ai+n1, aj+n1), ldda, batchCount, queue );

            // panel
            arginfo = magma_zpotrf_lpout_batched(uplo, n2, dAarray(ai+n1, aj+n1), ldda, gbstep + n1, info_array, batchCount, queue);
        }
    }
    return arginfo;

#undef dAarray
}
