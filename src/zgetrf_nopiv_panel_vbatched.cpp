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

// always assume for every matrix m >= n
// then max_minmn = max_n
magma_int_t
magma_zgetrf_nopiv_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue)
{
#define dA_array(i,j)    dA_array, i, j

    if( max_n <= min_recpnb ) {
        magma_zgetf2_nopiv_vbatched(
            m, n, minmn,
            max_m, max_n, max_minmn, max_mxn,
            dA_array, Ai, Aj, ldda,
            dtol_array, eps, info_array,
            gbstep, batchCount, queue);
    }
    else {
        magma_int_t max_n1 = max( min_recpnb, max_n / 2);
        magma_int_t max_n2 = max_n - max_n1;
        magma_int_t new_max_minmn = max_m * max_n1;
        // panel
        magma_zgetrf_nopiv_recpanel_vbatched(
            m, n, minmn,
            max_m, max_n1, max_n1, new_max_minmn, min_recpnb,
            dA_array(Ai, Aj), ldda,
            dtol_array, eps,
            info_array, gbstep, batchCount, queue);

        // trsm
        magmablas_ztrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            max_n1, max_n2, m, n, MAGMA_Z_ONE,
            dA_array(Ai, Aj       ), ldda,
            dA_array(Ai, Aj+max_n1), ldda,
            batchCount, queue );

        // gemm
        magmablas_zgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-max_n1, max_n2, max_n1,
            m, n, minmn,
            MAGMA_Z_NEG_ONE, dA_array(Ai+max_n1, Aj       ), ldda,
                             dA_array(Ai       , Aj+max_n1), ldda,
            MAGMA_Z_ONE,     dA_array(Ai+max_n1, Aj+max_n1), ldda,
            batchCount, queue );

        // panel 2
        new_max_minmn = (max_m-max_n1) * max_n2;
        magma_zgetrf_nopiv_recpanel_vbatched(
            m, n, minmn,
            max_m-max_n1, max_n2, max_n2, new_max_minmn, min_recpnb,
            dA_array(Ai+max_n1, Aj+max_n1), ldda,
            dtol_array, eps,
            info_array, gbstep+max_n1, batchCount, queue);
    }

    return 0;

#undef dA_array
}
