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

#define DBG
// always assume for every matrix m >= n
// then max_minmn = max_n
magma_int_t
magma_zgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue)
{
#define dA_array(i,j)    dA_array, i, j
#define dipiv_array(i)   dipiv_array, i

#ifdef DBG
    magma_int_t *h_M, *h_N, *h_ldda;
    magmaDoubleComplex **hA_array;
    magma_malloc_cpu((void**)&hA_array, batchCount * sizeof(magmaDoubleComplex*));
    magma_imalloc_cpu(&h_M,     batchCount);
    magma_imalloc_cpu(&h_N,     batchCount);
    magma_imalloc_cpu(&h_lddaa, batchCount);
    magma_getvector( batchCount, sizeof(magmaDoubleComplex), dA_array, 1, hA_array, 1, queue );
    magma_igetvector( batchCount, m,    1, h_M,    1, queue );
    magma_igetvector( batchCount, n,    1, h_N,    1, queue );
    magma_igetvector( batchCount, ldda, 1, h_ldda, 1, queue );
#endif

    if( max_n <= min_recpnb ) {
        magma_zgetf2_vbatched(
            m, n, minmn,
            max_m, max_n, max_minmn,
            dA_array, Ai, Aj, ldda,
            dipiv_array, info_array,
            gbstep, batchCount, queue);
    }
    else {
        magma_int_t max_n1 = max_n / 2;
        magma_int_t max_n2 = max_n - max_n1;

        // panel
        magma_zgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m, max_n1, max_n1, 0, min_recpnb,
            dA_array(Ai, Aj), ldda,
            dipiv_array, Ai, NULL,
            info_array, gbstep, batchCount, queue);
        #ifdef DBG
        printf("panel 1");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif

        // swap right
        magma_zlaswp_right_rowserial_vbatched(
            max_n2,
            m, n,
            dA_array(Ai, Aj+max_n1), ldda,
            Ai, Ai+max_n1,
            dipiv_array, batchCount, queue);
        #ifdef DBG
        printf("swap right");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif

        // trsm
        magmablas_ztrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            max_n1, max_n2, m, n, MAGMA_Z_ONE,
            dA_array(Ai, Aj       ), ldda,
            dA_array(Ai, Aj+max_n1), ldda,
            batchCount, queue );
        #ifdef DBG
        printf("trsm");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif

        // gemm
        magmablas_zgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-max_n1, max_n2, max_n1,
            m, n, minmn,
            MAGMA_Z_NEG_ONE, dA_array(Ai+max_n1, Aj       ), ldda,
                             dA_array(Ai       , Aj+max_n1), ldda,
            MAGMA_Z_ONE,     dA_array(Ai+max_n1, Aj+max_n1), ldda,
            batchCount, queue );
        #ifdef DBG
        printf("gemm");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif

        // panel 2
        magma_zgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m-max_n1, max_n2, max_n2, 0, min_recpnb,
            dA_array(Ai+max_n1, Aj+max_n1), ldda,
            dipiv_array, Ai+max_n1, NULL,
            info_array, gbstep+max_n1, batchCount, queue);
        #ifdef DBG
        printf("panel 2");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif

        // swap left
        magma_zlaswp_left_rowserial_vbatched(
            max_n1, max_n2,
            m, n, dA_array(Ai+max_n1, Aj), ldda,
            Ai+max_n1, Ai+max_n,
            dipiv_array,
            batchCount, queue);
        #ifdef DBG
        printf("swap left");
        magma_zprint_gpu(h_M[0], h_N[0], hA_array[0], h_ldda[0], queue);
        #endif


    }

#ifdef DBG
    magma_free_cpu( hA_array );
    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_ldda );
#endif
    return 0;

    #undef dA_array
    #undef dipiv_array
}
