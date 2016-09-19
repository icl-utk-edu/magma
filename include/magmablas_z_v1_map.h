/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef MAGMABLAS_Z_V1_MAP_H
#define MAGMABLAS_Z_V1_MAP_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define magmablas_ztranspose_inplace        magmablas_ztranspose_inplace_v1
#define magmablas_ztranspose_conj_inplace   magmablas_ztranspose_conj_inplace_v1
#define magmablas_ztranspose                magmablas_ztranspose_v1
#define magmablas_ztranspose_conj           magmablas_ztranspose_conj_v1
#define magmablas_zgetmatrix_transpose      magmablas_zgetmatrix_transpose_v1
#define magmablas_zsetmatrix_transpose      magmablas_zsetmatrix_transpose_v1
#define magmablas_zprbt                     magmablas_zprbt_v1
#define magmablas_zprbt_mv                  magmablas_zprbt_mv_v1
#define magmablas_zprbt_mtv                 magmablas_zprbt_mtv_v1
#define magma_zgetmatrix_1D_col_bcyclic     magma_zgetmatrix_1D_col_bcyclic_v1
#define magma_zsetmatrix_1D_col_bcyclic     magma_zsetmatrix_1D_col_bcyclic_v1
#define magma_zgetmatrix_1D_row_bcyclic     magma_zgetmatrix_1D_row_bcyclic_v1
#define magma_zsetmatrix_1D_row_bcyclic     magma_zsetmatrix_1D_row_bcyclic_v1
#define magmablas_zgeadd                    magmablas_zgeadd_v1
#define magmablas_zgeadd2                   magmablas_zgeadd2_v1
#define magmablas_zlacpy                    magmablas_zlacpy_v1
#define magmablas_zlacpy_conj               magmablas_zlacpy_conj_v1
#define magmablas_zlacpy_sym_in             magmablas_zlacpy_sym_in_v1
#define magmablas_zlacpy_sym_out            magmablas_zlacpy_sym_out_v1
#define magmablas_zlange                    magmablas_zlange_v1
#define magmablas_zlanhe                    magmablas_zlanhe_v1
#define magmablas_zlansy                    magmablas_zlansy_v1
#define magmablas_zlarfg                    magmablas_zlarfg_v1
#define magmablas_zlascl                    magmablas_zlascl_v1
#define magmablas_zlascl_2x2                magmablas_zlascl_2x2_v1
#define magmablas_zlascl2                   magmablas_zlascl2_v1
#define magmablas_zlascl_diag               magmablas_zlascl_diag_v1
#define magmablas_zlaset                    magmablas_zlaset_v1
#define magmablas_zlaset_band               magmablas_zlaset_band_v1
#define magmablas_zlaswp                    magmablas_zlaswp_v1
#define magmablas_zlaswp2                   magmablas_zlaswp2_v1
#define magmablas_zlaswp_sym                magmablas_zlaswp_sym_v1
#define magmablas_zlaswpx                   magmablas_zlaswpx_v1
#define magmablas_zsymmetrize               magmablas_zsymmetrize_v1
#define magmablas_zsymmetrize_tiles         magmablas_zsymmetrize_tiles_v1
#define magmablas_ztrtri_diag               magmablas_ztrtri_diag_v1
#define magmablas_dznrm2_adjust             magmablas_dznrm2_adjust_v1
#define magmablas_dznrm2_check              magmablas_dznrm2_check_v1
#define magmablas_dznrm2_cols               magmablas_dznrm2_cols_v1
#define magmablas_dznrm2_row_check_adjust   magmablas_dznrm2_row_check_adjust_v1
#define magma_zlarfb_gpu                    magma_zlarfb_gpu_v1
#define magma_zlarfb_gpu_gemm               magma_zlarfb_gpu_gemm_v1
#define magma_zlarfbx_gpu                   magma_zlarfbx_gpu_v1
#define magma_zlarfg_gpu                    magma_zlarfg_gpu_v1
#define magma_zlarfgtx_gpu                  magma_zlarfgtx_gpu_v1
#define magma_zlarfgx_gpu                   magma_zlarfgx_gpu_v1
#define magma_zlarfx_gpu                    magma_zlarfx_gpu_v1
#define magmablas_zaxpycp                   magmablas_zaxpycp_v1
#define magmablas_zswap                     magmablas_zswap_v1
#define magmablas_zswapblk                  magmablas_zswapblk_v1
#define magmablas_zswapdblk                 magmablas_zswapdblk_v1
#define magmablas_zgemv                     magmablas_zgemv_v1
#define magmablas_zgemv_conj                magmablas_zgemv_conj_v1
#define magmablas_zhemv                     magmablas_zhemv_v1
#define magmablas_zsymv                     magmablas_zsymv_v1
#define magmablas_zgemm                     magmablas_zgemm_v1
#define magmablas_zgemm_reduce              magmablas_zgemm_reduce_v1
#define magmablas_zhemm                     magmablas_zhemm_v1
#define magmablas_zsymm                     magmablas_zsymm_v1
#define magmablas_zsyr2k                    magmablas_zsyr2k_v1
#define magmablas_zher2k                    magmablas_zher2k_v1
#define magmablas_zsyrk                     magmablas_zsyrk_v1
#define magmablas_zherk                     magmablas_zherk_v1
#define magmablas_ztrsm                     magmablas_ztrsm_v1
#define magmablas_ztrsm_outofplace          magmablas_ztrsm_outofplace_v1
#define magmablas_ztrsm_work                magmablas_ztrsm_work_v1

#undef magma_zsetvector
#undef magma_zgetvector
#undef magma_zcopyvector
#undef magma_zsetmatrix
#undef magma_zgetmatrix
#undef magma_zcopymatrix

#define magma_zsetvector                    magma_zsetvector_v1
#define magma_zgetvector                    magma_zgetvector_v1
#define magma_zcopyvector                   magma_zcopyvector_v1
#define magma_zsetmatrix                    magma_zsetmatrix_v1
#define magma_zgetmatrix                    magma_zgetmatrix_v1
#define magma_zcopymatrix                   magma_zcopymatrix_v1

#define magma_izamax                        magma_izamax_v1
#define magma_izamin                        magma_izamin_v1
#define magma_dzasum                        magma_dzasum_v1
#define magma_zaxpy                         magma_zaxpy_v1
#define magma_zcopy                         magma_zcopy_v1
#define magma_zdotc                         magma_zdotc_v1
#define magma_zdotu                         magma_zdotu_v1
#define magma_dznrm2                        magma_dznrm2_v1
#define magma_zrot                          magma_zrot_v1
#define magma_zdrot                         magma_zdrot_v1
#define magma_zrotm                         magma_zrotm_v1
#define magma_zrotmg                        magma_zrotmg_v1
#define magma_zscal                         magma_zscal_v1
#define magma_zdscal                        magma_zdscal_v1
#define magma_zswap                         magma_zswap_v1
#define magma_zgemv                         magma_zgemv_v1
#define magma_zgerc                         magma_zgerc_v1
#define magma_zgeru                         magma_zgeru_v1
#define magma_zhemv                         magma_zhemv_v1
#define magma_zher                          magma_zher_v1
#define magma_zher2                         magma_zher2_v1
#define magma_ztrmv                         magma_ztrmv_v1
#define magma_ztrsv                         magma_ztrsv_v1
#define magma_zgemm                         magma_zgemm_v1
#define magma_zsymm                         magma_zsymm_v1
#define magma_zhemm                         magma_zhemm_v1
#define magma_zsyr2k                        magma_zsyr2k_v1
#define magma_zher2k                        magma_zher2k_v1
#define magma_zsyrk                         magma_zsyrk_v1
#define magma_zherk                         magma_zherk_v1
#define magma_ztrmm                         magma_ztrmm_v1
#define magma_ztrsm                         magma_ztrsm_v1

#endif // MAGMABLAS_Z_V1_MAP_H
