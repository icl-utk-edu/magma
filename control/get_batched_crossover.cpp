/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Definition of blocking sizes for NVIDIA cards
#ifdef HAVE_CUBLAS

// =============================================================================
/// @addtogroup magma_tuning
/// @{

#define ZPOTRF_SWITCH 160
#define CPOTRF_SWITCH 224
#define DPOTRF_SWITCH 384
#define SPOTRF_SWITCH 432

#define ZPOTRF_VBATCHED_SWITCH 448
#define CPOTRF_VBATCHED_SWITCH 384
#define DPOTRF_VBATCHED_SWITCH 480
#define SPOTRF_VBATCHED_SWITCH 704

/***************************************************************************//**
    Returns in nb and recnb the crossover points for potrf based on n
*******************************************************************************/
void magma_get_zpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= ZPOTRF_SWITCH)
    {
        *nb    = ZPOTRF_SWITCH;
        *recnb = ZPOTRF_SWITCH;
        return;
    }
    *nb    = 64;
    *recnb = 32;
    return;
}

/// @see magma_get_zpotrf_batched_nbparam
void magma_get_cpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= CPOTRF_SWITCH)
    {
        *nb    = CPOTRF_SWITCH;
        *recnb = CPOTRF_SWITCH;
        return;
    }
    
    if (n <= 256)
    {
        *nb    = 256;
        *recnb = 256;
    }
    else {
        *nb    = 128;
        *recnb =  32;
    }
    return;
}

/// @see magma_get_zpotrf_batched_nbparam
void magma_get_dpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= DPOTRF_SWITCH)
    {
        *nb    = DPOTRF_SWITCH;
        *recnb = DPOTRF_SWITCH;
        return;
    }
    if (n <= 384)
    {
        *nb    = 384;
        *recnb = 384;
    }
    else {
        *nb    = 128;
        *recnb =  32;
    }
    return;
}

/// @see magma_get_zpotrf_batched_nbparam
void magma_get_spotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= SPOTRF_SWITCH)
    {
        *nb    = SPOTRF_SWITCH;
        *recnb = SPOTRF_SWITCH;
        return;
    }
    if (n <= 464)
    {
        *nb    = 512;
        *recnb = 512;
    }
    else {
        *nb    = 256;
        *recnb =  64;
    }
    return;
}


/***************************************************************************//**
    Returns in nb and recnb the crossover points for potrf based on m
*******************************************************************************/
void magma_get_zgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 64;
    *recnb = 32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_cgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_dgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_sgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}


/***************************************************************************//**
    @return nb for geqrf_batched based on n
*******************************************************************************/
// TODO: get_geqrf_nb takes (m,n); this should do likewise
magma_int_t magma_get_zgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

/// @see magma_get_zgeqrf_batched_nb
magma_int_t magma_get_cgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

/// @see magma_get_zgeqrf_batched_nb
magma_int_t magma_get_dgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

/// @see magma_get_zgeqrf_batched_nb
magma_int_t magma_get_sgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}


/***************************************************************************//**
    @return the crossover point between the _lg or the kernel directly
*******************************************************************************/
magma_int_t magma_get_zpotrf_batched_crossover()
{
    magma_int_t arch = magma_getdevice_arch();
    if(arch >= 700){
        return 352;
    }
    else if(arch >= 600){
        return 352;
    }
    else{
        return 160;
    }
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_cpotrf_batched_crossover()
{
    magma_int_t arch = magma_getdevice_arch();
    if(arch >= 700){
        return 576;
    }
    else if(arch >= 600){
        return 544;
    }
    else{
        return 224;
    }
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_dpotrf_batched_crossover()
{
    magma_int_t arch = magma_getdevice_arch();
    if(arch >= 700){
        return 640;
    }
    else if(arch >= 600){
        return 576;
    }
    else{
        return 384;
    }
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_spotrf_batched_crossover()
{
    magma_int_t arch = magma_getdevice_arch();
    if(arch >= 700){
        return 608;
    }
    else if(arch >= 600){
        return 544;
    }
    else{
        return 432;
    }
}
/***************************************************************************//**
    @return the crossover point between the _lg or the kernel directly
*******************************************************************************/
magma_int_t magma_get_zpotrf_vbatched_crossover()
{
    return ZPOTRF_VBATCHED_SWITCH;
}

/// @see magma_get_zpotrf_vbatched_crossover
magma_int_t magma_get_cpotrf_vbatched_crossover()
{
    return CPOTRF_VBATCHED_SWITCH;
}

/// @see magma_get_zpotrf_vbatched_crossover
magma_int_t magma_get_dpotrf_vbatched_crossover()
{
    return DPOTRF_VBATCHED_SWITCH;
}

/// @see magma_get_zpotrf_vbatched_crossover
magma_int_t magma_get_spotrf_vbatched_crossover()
{
    return SPOTRF_VBATCHED_SWITCH;
}


/***************************************************************************//**
    @return the ntcol value for very small xgetri_batched ( m = n )
*******************************************************************************/
magma_int_t magma_get_zgetri_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t ntcol = 1;
    
    // TODO: conduct tuning experiment for ntcol in z precision
    if(m == n){
        if( m < 16) 
            ntcol =  2;
        else
            ntcol = 1;
    }
    return ntcol;
}

/// @see magma_get_zgetri_batched_ntcol
magma_int_t magma_get_cgetri_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t ntcol = 1;
    
    // TODO: conduct tuning experiment for ntcol in z precision
    if(m == n){
        if( m < 16) 
            ntcol =  2;
        else
            ntcol = 1;
    }
    return ntcol;
}

/// @see magma_get_zgetri_batched_ntcol
magma_int_t magma_get_dgetri_batched_ntcol(magma_int_t m, magma_int_t n)
{
    
    // TODO: conduct tuning experiment for ntcol on Kepler
    magma_int_t arch = magma_getdevice_arch();
    magma_int_t ntcol = 1;
    if(m == n ){
        switch(m){
            case  1: ntcol = (arch >= 600) ?  8: 32 ; break;
            case  2: ntcol = (arch >= 600) ?  4: 16 ; break;
            case  3: ntcol = (arch >= 600) ?  3:  8 ; break;
            case  4: ntcol = (arch >= 600) ?  3: 16 ; break;
            case  5: ntcol = (arch >= 600) ?  2:  8 ; break;
            case  6: ntcol = (arch >= 600) ?  2:  4 ; break;
            case  7: ntcol = (arch >= 600) ?  2:  4 ; break;
            case  8: ntcol = (arch >= 600) ? 32: 32 ; break;
            case  9: ntcol = (arch >= 600) ? 16:  4 ; break;
            case 10: ntcol = (arch >= 600) ? 14:  4 ; break;
            case 11: ntcol = (arch >= 600) ? 10:  8 ; break;
            case 12: ntcol = (arch >= 600) ? 12:  8 ; break;
            case 13: ntcol = (arch >= 600) ? 12:  8 ; break;
            case 14: ntcol = (arch >= 600) ? 14:  8 ; break;
            case 15: ntcol = (arch >= 600) ?  8:  8 ; break;
            case 16: ntcol = (arch >= 600) ? 10:  8 ; break;
            case 17: ntcol = (arch >= 600) ?  5:  4 ; break;
            case 18: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 19: ntcol = (arch >= 600) ?  4: 16 ; break;
            case 20: ntcol = (arch >= 600) ?  4: 16 ; break;
            case 21: ntcol = (arch >= 600) ?  4: 16 ; break;
            case 22: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 23: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 24: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 25: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 26: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 27: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 28: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 29: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 30: ntcol = (arch >= 600) ?  4:  4 ; break;
            case 31: ntcol = (arch >= 600) ?  2:  4 ; break;
            case 32: ntcol = (arch >= 600) ?  2:  4 ; break;
            default: ntcol = 1;
        }
    }
    return ntcol;
}

/// @see magma_get_zgetri_batched_ntcol
magma_int_t magma_get_sgetri_batched_ntcol(magma_int_t m, magma_int_t n)
{
    // TODO: conduct tuning experiment for ntcol on Kepler
    magma_int_t arch = magma_getdevice_arch();
    magma_int_t ntcol = 1;
    if(m == n ){
        switch(m){
            case  1: ntcol = (arch >= 600) ?  9 : 32 ; break;
            case  2: ntcol = (arch >= 600) ?  4 : 16 ; break;
            case  3: ntcol = (arch >= 600) ?  3 :  8 ; break;
            case  4: ntcol = (arch >= 600) ?  4 :  8 ; break;
            case  5: ntcol = (arch >= 600) ?  4 :  8 ; break;
            case  6: ntcol = (arch >= 600) ?  3 :  8 ; break;
            case  7: ntcol = (arch >= 600) ?  3 :  8 ; break;
            case  8: ntcol = (arch >= 600) ? 14 : 32 ; break;
            case  9: ntcol = (arch >= 600) ? 16 :  8 ; break;
            case 10: ntcol = (arch >= 600) ? 16 : 16 ; break;
            case 11: ntcol = (arch >= 600) ? 32 :  8 ; break;
            case 12: ntcol = (arch >= 600) ? 32 :  8 ; break;
            case 13: ntcol = (arch >= 600) ? 32 :  8 ; break;
            case 14: ntcol = (arch >= 600) ? 16 :  8 ; break;
            case 15: ntcol = (arch >= 600) ? 14 :  8 ; break;
            case 16: ntcol = (arch >= 600) ? 16 :  8 ; break;
            case 17: ntcol = (arch >= 600) ?  9 :  4 ; break;
            case 18: ntcol = (arch >= 600) ?  9 :  4 ; break;
            case 19: ntcol = (arch >= 600) ?  9 :  4 ; break;
            case 20: ntcol = (arch >= 600) ?  8 :  8 ; break;
            case 21: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 22: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 23: ntcol = (arch >= 600) ?  8 :  4 ; break;
            case 24: ntcol = (arch >= 600) ?  8 :  4 ; break;
            case 25: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 26: ntcol = (arch >= 600) ?  4 :  8 ; break;
            case 27: ntcol = (arch >= 600) ?  4 :  8 ; break;
            case 28: ntcol = (arch >= 600) ?  4 :  8 ; break;
            case 29: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 30: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 31: ntcol = (arch >= 600) ?  4 :  4 ; break;
            case 32: ntcol = (arch >= 600) ?  4 :  4 ; break;
            default: ntcol = 1;
        }
    }
    return ntcol;
}

/***************************************************************************//**
    @return the stop nb value for recursive batched trsm
*******************************************************************************/
magma_int_t magma_get_ztrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n)
{
    if(side == MagmaLeft){
         if     (m <= 2) return 2; 
         else if(m <= 4) return 4;
         else if(m <= 8) return 8;
         else{
             if(n <= 32) return 16;
             else return 8; 
         }
    }else{    // side = MagmaRight
        if(n <= 2) return 2;
        else return 8;
    }
} 

/// @see magma_get_ztrsm_batched_stop_nb
magma_int_t magma_get_ctrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n)
{
    if(side == MagmaLeft){
        if(m <= 8) return 8;
        else return 16;
    }else{    // side = MagmaRight
        if(n <= 4) return 4;
        else return 16;
    }
} 

/// @see magma_get_ztrsm_batched_stop_nb
magma_int_t magma_get_dtrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n)
{
    if(side == MagmaLeft){
        if     (m <= 2) return 8;
        else if(m <= 4) return 16;
        else return 32;
    }else{    // side = MagmaRight
        if(n <= 4) return 4;
        else return 32;
    }
} 

/// @see magma_get_ztrsm_batched_stop_nb
magma_int_t magma_get_strsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n)
{
    if(side == MagmaLeft){
        return 16;
    }else{    // side = MagmaRight
        if     (n <= 4) return 4;
        else if(n <= 8) return 8;
        else return 32;
    }
} 

// =============================================================================
/// @}
// end group magma_tuning

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
