/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/
#ifndef BATCHED_KERNEL_PARAM_H
#define BATCHED_KERNEL_PARAM_H

#define MAX_NTHREADS        1024     // 1024 is max threads for 2.x cards
#define MAX_SHARED_ALLOWED    47

#define zamax            256
#define zamax_shuffle    256
#define DOTC_MAX_BS      512     // 512 is max threads for 1.x cards


#define POTRF_NB         128     // blocking in main algorithm 128 if using recursive panel or 32 if using standard panel
#define POTF2_NB           8     // blocking size in panel factorization
#define POTF2_NNB          8  // blocking size in panel factorization
#define POTF2_TILE_SIZE   32
#define MAX_POTF2_SM     128
#define VERSION20

#define BATRF_NB         128
#define BATRF_RECNB       32
#define BATF2_NB           8
#define BASWP_WIDTH        4
#define SWP_WIDTH          4

#define BAQRF_NB         32

#define BATRI_NB         128        // ztrsm_nb should be >= BATRF_NB
#define TRI_NB           128        // ztrsm_nb should match the NB in BATRF_NB
#define TRI_BLOCK_SIZE    16

// tuning for zgetf2_panel
#define ZGETF2_PANEL_NB    (8)
#define CGETF2_PANEL_NB    (8)
#define DGETF2_PANEL_NB    (16)
#define SGETF2_PANEL_NB    (16)

// tuning zgetf2_panel_chain
#define ZGETF2_FUSED_NTH    (512)
#define CGETF2_FUSED_NTH    (512)
#define DGETF2_FUSED_NTH    (512)
#define SGETF2_FUSED_NTH    (512)
//
#define ZGETF2_FUSED_MAX_M  (7168)
#define CGETF2_FUSED_MAX_M  (10240)
#define DGETF2_FUSED_MAX_M  (25600)
#define SGETF2_FUSED_MAX_M  (40960)



// TRSM tuning parameters 
#define STRTRI_BATCHED_NB         (64)
#define STRTRI_BATCHED_BLOCK_SIZE (16)
#define DTRTRI_BATCHED_NB         (64)
#define DTRTRI_BATCHED_BLOCK_SIZE (16)
#define CTRTRI_BATCHED_NB         (32)
#define CTRTRI_BATCHED_BLOCK_SIZE (16)
#define ZTRTRI_BATCHED_NB         (128)
#define ZTRTRI_BATCHED_BLOCK_SIZE (16)

// HEMM tuning 
#define ZHEMM_BATCHED_LEFT    8, 16, 16, 1
#define ZHEMM_BATCHED_RIGHT   8, 16, 16, 1
#define CHEMM_BATCHED_LEFT    16, 32, 32, 1
#define CHEMM_BATCHED_RIGHT   16, 32, 32, 1
#define DSYMM_BATCHED_LEFT    16, 32, 32, 0
#define DSYMM_BATCHED_RIGHT   16, 32, 32, 0
#define SSYMM_BATCHED_LEFT    32, 64, 64, 0
#define SSYMM_BATCHED_RIGHT   32, 64, 64, 0

// TRMM tuning
#define ZTRMM_BATCHED_NB    (16)
#define CTRMM_BATCHED_NB    (16)
#define DTRMM_BATCHED_NB    (32)
#define STRMM_BATCHED_NB    (32)

// HEMV tuning
#define ZHEMV_BATCHED_LOWER    16, 4
#define CHEMV_BATCHED_LOWER    16, 4
#define DSYMV_BATCHED_LOWER    16, 4
#define SSYMV_BATCHED_LOWER    32, 4
#define ZHEMV_BATCHED_UPPER    16, 4
#define CHEMV_BATCHED_UPPER    16, 4
#define DSYMV_BATCHED_UPPER    16, 4
#define SSYMV_BATCHED_UPPER    32, 4

// GETF2_FUSED_BATCHED maximum rows
#define ZGETF2_FUSED_BATCHED_MAX_ROWS    (256)
#define CGETF2_FUSED_BATCHED_MAX_ROWS    (384)
#define DGETF2_FUSED_BATCHED_MAX_ROWS    (512)
#define SGETF2_FUSED_BATCHED_MAX_ROWS    (512)

#define magma_ceilpow2(N)    ( (N >  16)? 32 : \
                               (N >   8)? 16 : \
                               (N >   4)?  8 : \
                               (N >   2)?  4 : \
                               (N >   0)?  2 : 0 )    

#endif        //  #ifndef BATCHED_KERNEL_PARAM_H
