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

magma_int_t
magma_zgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t max_minmn, magma_int_t max_mxn, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue)
{
#define dAarray(i,j)    dA_array, i, j
#define ipiv_array(i)   dipiv_array, i


    magma_int_t *minmn;
    magma_imalloc(&minmn, batchCount);
    magma_ivec_min_vv( batchCount, m, n, minmn, queue);

    magma_zgetf2_vbatched(
        m, n, minmn,
        max_m, max_n, max_minmn,
        dA_array, Ai, Aj, ldda,
        dipiv_array, info_array,
        0, batchCount, queue);

    magma_free( minmn );

    return 0;

    #undef dAarray
    #undef ipiv_array
}
