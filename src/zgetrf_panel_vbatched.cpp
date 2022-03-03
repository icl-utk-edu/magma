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

//#define DBG
#define ZGETRF2_VBATCHED_PAR_SWAP

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

#if defined(DBG) && defined(PRECISION_d)
    const int ib = 3;
    magmaDoubleComplex** hA_array=NULL;
    magma_int_t **hpiv_array, **hpiv2_array;
    magma_int_t *hM=NULL, *hN=NULL, *hldda=NULL;
    magma_malloc_cpu((void**)&hA_array, batchCount*sizeof(magmaDoubleComplex*));
    magma_malloc_cpu((void**)&hpiv_array,  batchCount*sizeof(magma_int_t*));
    magma_malloc_cpu((void**)&hpiv2_array, batchCount*sizeof(magma_int_t*));
    magma_imalloc_cpu(&hM, batchCount);
    magma_imalloc_cpu(&hN, batchCount);
    magma_imalloc_cpu(&hldda, batchCount);
    magma_getvector(batchCount, sizeof(magmaDoubleComplex*), dA_array, 1, hA_array, 1, queue);
    magma_getvector(batchCount, sizeof(magma_int_t*), dipiv_array,  1, hpiv_array, 1, queue);
    magma_getvector(batchCount, sizeof(magma_int_t*), dpivinfo_array, 1, hpiv2_array, 1, queue);
    magma_igetvector(batchCount, m, 1, hM, 1, queue);
    magma_igetvector(batchCount, n, 1, hN, 1, queue);
    magma_igetvector(batchCount, ldda, 1, hldda, 1, queue);
    #endif

#if defined(DBG) && defined(PRECISION_d)
    printf("org -- (Ai, Aj) = (%d, %d)\n", Ai, Aj);
    const int iminmn = min(hM[ib], hN[ib]);
    magma_int_t* hpiv  = new magma_int_t[iminmn];
    magma_int_t* hpiv2 = new magma_int_t[hM[ib]];
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
#endif

    if( max_n <= min_recpnb ) {
        #if defined(DBG) && defined(PRECISION_d)
        printf("stop condition -- (Ai, Aj) = (%d, %d)\n", Ai, Aj);
        #endif
        magma_zgetf2_vbatched(
            m, n, minmn,
            max_m, max_n, max_minmn, max_mxn,
            dA_array, Ai, Aj, ldda,
            dipiv_array, info_array,
            gbstep, batchCount, queue);
    }
    else {
        magma_int_t max_n1 = max( min_recpnb, max_n / 2);
        magma_int_t max_n2 = max_n - max_n1;
        magma_int_t new_max_minmn = max_m * max_n1;
        // panel
        magma_zgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m, max_n1, max_n1, new_max_minmn, min_recpnb,
            dA_array(Ai, Aj), ldda,
            dipiv_array, Ai, dpivinfo_array,
            info_array, gbstep, batchCount, queue);

#if defined(DBG) && defined(PRECISION_d)
    printf("panel -- (Ai, Aj) = (%d, %d)\n", Ai, Aj);
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif
        // swap right
        #ifdef ZGETRF2_VBATCHED_PAR_SWAP
        setup_pivinfo_vbatched(dpivinfo_array, Ai, dipiv_array, Ai, m, n, max_m, max_n1, batchCount, queue);
        magma_zlaswp_right_rowparallel_vbatched(
            max_n2,
            m, n,
            dA_array(Ai, Aj+max_n1), ldda,
            0, max_n1,
            dpivinfo_array, Ai,
            batchCount, queue);
        #else
        magma_zlaswp_right_rowserial_vbatched(
            max_n2,
            m, n,
            dA_array(Ai, Aj+max_n1), ldda,
            dipiv_array(Ai),
            0, max_n1,
            batchCount, queue);
        #endif

#if defined(DBG) && defined(PRECISION_d)
    printf("swap right\n");
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif
        // trsm
        magmablas_ztrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            max_n1, max_n2, m, n, MAGMA_Z_ONE,
            dA_array(Ai, Aj       ), ldda,
            dA_array(Ai, Aj+max_n1), ldda,
            batchCount, queue );

#if defined(DBG) && defined(PRECISION_d)
    printf("trsm\n");
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
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

#if defined(DBG) && defined(PRECISION_d)
    printf("gemm\n");
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif
        // panel 2
        new_max_minmn = (max_m-max_n1) * max_n2;
        magma_zgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m-max_n1, max_n2, max_n2, new_max_minmn, min_recpnb,
            dA_array(Ai+max_n1, Aj+max_n1), ldda,
            dipiv_array, Ai+max_n1, dpivinfo_array,
            info_array, gbstep+max_n1, batchCount, queue);

#if defined(DBG) && defined(PRECISION_d)
    printf("panel 2 -- offsets (%d, %d)\n", Ai+max_n1, Aj+max_n1);
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif
        // swap left
        #ifdef ZGETRF2_VBATCHED_PAR_SWAP
        setup_pivinfo_vbatched(dpivinfo_array, Ai+max_n1, dipiv_array, Ai+max_n1, m, n, max_m-max_n1, max_n2, batchCount, queue);
        magma_zlaswp_left_rowparallel_vbatched(
            max_n1,
            m, n, dA_array(Ai+max_n1, Aj), ldda,
            0, max_n2,
            dpivinfo_array, Ai+max_n1,
            batchCount, queue);
        #else
        magma_zlaswp_left_rowserial_vbatched(
            max_n1,
            m, n, dA_array(Ai+max_n1, Aj), ldda,
            dipiv_array(Ai+max_n1),
            0, max_n2,
            batchCount, queue);
        #endif
#if defined(DBG) && defined(PRECISION_d)
    printf("swap left\n");
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif

        // adjust pivot
        adjust_ipiv_vbatched(dipiv_array, Ai+max_n1, minmn, max_n2, max_n1, batchCount, queue);
    }

#if defined(DBG) && defined(PRECISION_d)
    printf("end\n");
    magma_dprint_gpu(hM[ib], hN[ib], hA_array[ib], hldda[ib], queue);
    magma_igetvector(iminmn, hpiv_array[ib], 1, hpiv, 1, queue);
    magma_igetvector(hM[ib], hpiv2_array[ib], 1, hpiv2, 1, queue);
    for(int s = 0; s < iminmn; s++){printf("piv[%2d]     = %2d\n", s, hpiv[s]);}
    for(int s = 0; s < hM[ib]; s++){printf("pivinfo[%2d] = %2d\n", s, hpiv2[s]);}
#endif

#if defined(DBG) && defined(PRECISION_d)
    magma_free_cpu(hA_array);
    magma_free_cpu(hM);
    magma_free_cpu(hN);
    magma_free_cpu(hldda);
    magma_free_cpu(hpiv_array);
    magma_free_cpu(hpiv2_array);

    delete[] hpiv;
    delete[] hpiv2;
#endif

    return 0;

#undef dA_array
#undef dipiv_array
}
