/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#include <vector>
#include <cmath>
#include "magma_internal.h"
#include "geqrf_batched_panel_decision.h"

#ifdef __cplusplus
extern "C" {
#endif

// Definition of blocking sizes for NVIDIA cards
// TODO: update for SYCL
#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP) || defined(MAGMA_HAVE_SYCL)

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
    Returns in nb and recnb the crossover points for getrf
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
    Returns in nb and recnb the crossover points for getrf
*******************************************************************************/
void magma_get_zgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = (max_m <= 192) ? 32 :
             (max_m <= 384) ? 64 : 128;
    *recnb = 32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_cgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = (max_m <= 192) ? 32 :
             (max_m <= 384) ? 64 : 128;
    *recnb =  32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_dgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = (max_m <= 192) ? 32 :
             (max_m <= 384) ? 64 : 128;
    *recnb =  32;
    return;
}

/// @see magma_get_zgetrf_batched_nbparam
void magma_get_sgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = (max_m <= 192) ? 32 :
             (max_m <= 384) ? 64 : 128;
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
    @return recommendation (1/0) of using the panel code only (with fused
            update) over the main blocked code
*******************************************************************************/
// this is a generic search routine for the lookup tables defined in
// geqrf_batched_panel_decision.h
#define GEQRF_BATCHED_LOOKUP_TABLE_BATCH_STEP   (100)
#define GEQRF_BATCHED_MAX_TESTED_WIDTH          (256)

static magma_int_t magma_geqrf_batched_get_cutoff_width(
            magma_int_t m, magma_int_t n, magma_int_t batchCount,
            std::vector<std::vector<magma_int_t>>* lookup_table )
{
    magma_int_t cutoff_width = 0;
    magma_int_t batch_index  = (magma_int_t) nearbyint( (double)batchCount / (double)GEQRF_BATCHED_LOOKUP_TABLE_BATCH_STEP );
    batch_index = (batch_index == 0) ? 1 : batch_index;  // the first column in the table is for 'm', not the cutoff-width
    size_t table_size   = (magma_int_t) lookup_table->size();
    size_t m_index = 0;
    // find the closest m
    magma_int_t dist = (magma_int_t)(INT_MAX);
    for(size_t i = 0; i < table_size; i++) {
        double idist = std::abs(m - (*lookup_table)[i][0]);
        if(idist < dist) {
            m_index = i;
            dist    = idist;
        }
    }

    // make sure we don't go out-of-bounds
    batch_index = min( batch_index, (magma_int_t)((*lookup_table)[m_index].size()-1) );

    cutoff_width = (*lookup_table)[m_index][batch_index];

    // if the cutoff_width is equal to the maximum tested width during the tuning sweeps,
    // this probably means to use the fused update even for larger widths
    cutoff_width = ( cutoff_width == GEQRF_BATCHED_MAX_TESTED_WIDTH ) ? n : cutoff_width;

    return cutoff_width;
}

magma_int_t magma_use_zgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
    magma_int_t use_fused_update = 0, cutoff_width = 0;
    std::vector<std::vector<magma_int_t>>* data;
    #ifdef MAGMA_HAVE_CUDA
    // TODO: add more gpus
    data = &zgeqrf_panel_decision_a100;
    #else
    // TODO: add more gpus
    data = &zgeqrf_panel_decision_a100;
    #endif

    cutoff_width     = magma_geqrf_batched_get_cutoff_width(m, n, batchCount, data);
    use_fused_update = (n <= cutoff_width) ? 1 : 0;
    return use_fused_update;
}

magma_int_t magma_use_cgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
    magma_int_t use_fused_update = 0, cutoff_width = 0;
    std::vector<std::vector<magma_int_t>>* data;
    #ifdef MAGMA_HAVE_CUDA
    // TODO: add more gpus
    data = &cgeqrf_panel_decision_a100;
    #else
    // TODO: add more gpus
    data = &cgeqrf_panel_decision_a100;
    #endif

    cutoff_width     = magma_geqrf_batched_get_cutoff_width(m, n, batchCount, data);
    use_fused_update = (n <= cutoff_width) ? 1 : 0;
    return use_fused_update;
}

magma_int_t magma_use_dgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
    magma_int_t use_fused_update = 0, cutoff_width = 0;
    std::vector<std::vector<magma_int_t>>* data;
    #ifdef MAGMA_HAVE_CUDA
    // TODO: add more gpus
    data = &dgeqrf_panel_decision_a100;
    #else
    // TODO: add more gpus
    data = &dgeqrf_panel_decision_a100;
    #endif

    cutoff_width     = magma_geqrf_batched_get_cutoff_width(m, n, batchCount, data);
    use_fused_update = (n <= cutoff_width) ? 1 : 0;
    return use_fused_update;
}

magma_int_t magma_use_sgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
    magma_int_t use_fused_update = 0, cutoff_width = 0;
    std::vector<std::vector<magma_int_t>>* data;
    #ifdef MAGMA_HAVE_CUDA
    // TODO: add more gpus
    data = &sgeqrf_panel_decision_a100;
    #else
    // TODO: add more gpus
    data = &sgeqrf_panel_decision_a100;
    #endif

    cutoff_width     = magma_geqrf_batched_get_cutoff_width(m, n, batchCount, data);
    use_fused_update = (n <= cutoff_width) ? 1 : 0;
    return use_fused_update;
}

/***************************************************************************//**
    @return the recommended #threads for geqr2_fused_sm_batched
*******************************************************************************/
magma_int_t magma_get_zgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n)
{
    #ifdef MAGMA_HAVE_HIP
    // based on MI100, rocm-4.5.0
    if ( n <= 4 ) {
        if      ( m <= 160 ) return  32;
        else if ( m <= 288 ) return  64;
        else if ( m <= 320 ) return  32;
        else if ( m <= 448 ) return 128;
        else if ( m <= 480 ) return  64;
        else if ( m <= 480 ) return  64;
        else                 return 128;
    }
    else if ( n <= 8 ) {
        if      ( m <= 160 ) return  32;
        else                 return 128;
    }
    else {
        return 128; // panel is wide, use a large number of threads
    }
    #else
    // based on A100, cuda-11.2.0
    if ( n <= 4 ) {
        if      ( m <= 224 ) return 32;
        else if ( m <= 480 ) return 64;
        else if ( m <= 800 ) return 128;
        else                 return 256;
    }
    else if ( n <= 8 ) {
        if      ( m <=  96 ) return 32;
        else if ( m <= 224 ) return 64;
        else if ( m <= 608 ) return 128;
        else                 return 256;
    }
    else {
        return 512; // panel is wide, use a large number of threads
    }
    #endif
}

magma_int_t magma_get_cgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n)
{
    #ifdef MAGMA_HAVE_HIP
    // based on MI100, rocm-4.5.0
    if ( n <= 4 ) {
        if      ( m <= 192 ) return  32;
        else if ( m <= 352 ) return  64;
        else if ( m <= 384 ) return  32;
        else if ( m <= 608 ) return 128;
        else if ( m <= 640 ) return  64;
        else                 return 128;
    }
    else if ( n <= 8 ) {
        if      ( m <= 192 ) return  32;
        else if ( m <= 288 ) return 128;
        else if ( m <= 320 ) return  64;
        else if ( m <= 640 ) return 128;
        else                 return 256;
    }
    else {
        return 512; // panel is wide, use a large number of threads
    }
    #else
    // based on A100, cuda-11.2.0
    if ( n <= 4 ) {
        if      ( m <= 288 ) return 32;
        else if ( m <= 448 ) return 64;
        else if ( m <= 960 ) return 128;
        else                 return 256;
    }
    else if ( n <= 8 ) {
        if      ( m <=  160 ) return 32;
        else if ( m <=  256 ) return 64;
        else if ( m <=  608 ) return 128;
        else                  return 256;
    }
    else if (n <= 16) {
        if      ( m <=  288 ) return 128;
        else if ( m <=  608 ) return 256;
        else                  return 512;
    }
    else {
        return 512; // panel is too wide, use a large number of threads
    }
    #endif
}

magma_int_t magma_get_dgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n)
{
    #ifdef MAGMA_HAVE_HIP
    // based on MI100, rocm-4.5.0
    if ( n <= 4 ) {
        if      ( m <= 192 ) return  32;
        else if ( m <= 352 ) return  64;
        else if ( m <= 384 ) return  32;
        else if ( m <= 608 ) return 128;
        else if ( m <= 640 ) return  64;
        else                 return 128;
    }
    else if ( n <= 8 ) {
        if      ( m <= 192 ) return  32;
        else if ( m <= 320 ) return  64;
        else if ( m <= 640 ) return 128;
        else                 return 256;
    }
    else {
        return 512; // panel is wide, use a large number of threads
    }
    #else
    // based on A100, cuda-11.2.0
    if ( n <= 4 ) {
        if      ( m <=  224 ) return 32;
        else if ( m <=  448 ) return 64;
        else if ( m <=  960 ) return 128;
        else                  return 256;
    }
    else if ( n <= 8 ) {
        if      ( m <=  160 ) return 32;
        else if ( m <=  256 ) return 64;
        else if ( m <=  608 ) return 128;
        else                  return 256;
    }
    else if ( n <= 16 ) {
        if      ( m <=  224 ) return 128;
        else if ( m <=  608 ) return 256;
        else                  return 512;
    }
    else {
        return 512; // panel is too wide, use a large number of threads
    }
    #endif
}

magma_int_t magma_get_sgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n)
{
    #ifdef MAGMA_HAVE_HIP
    // based on MI100, rocm-4.5.0
    if ( n <= 4 ) {
        if      ( m <= 192 ) return  32;
        else if ( m <= 448 ) return  64;
        else if ( m <= 736 ) return 128;
        else if ( m <= 768 ) return  64;
        else                 return 128;
    }
    else if ( n <= 8 ) {
        if      ( m <=  384 ) return  64;
        else if ( m <=  640 ) return 128;
        else if ( m <=  960 ) return 256;
        else if ( m <= 1024 ) return 128;
        else                  return 256;
    }
    else {
        return 512; // panel is wide, use a large number of threads
    }
    #else
    // based on A100, cuda-11.2.0
    if ( n <= 4 ) {
        if      ( m <=  192 ) return 32;
        else if ( m <=  960 ) return 64;
        else                  return 128;
    }
    else if ( n <= 8 ) {
        if      ( m <=  160 ) return 32;
        else if ( m <=  480 ) return 64;
        else if ( m <=  992 ) return 128;
        else                  return 256;
    }
    else if ( n <= 16 ) {
        if      ( m <=  224 ) return 64;
        else if ( m <=  480 ) return 128;
        else if ( m <= 1024 ) return 256;
        else                  return 512;
    }
    else {
        return 512; // panel is too wide, use a large number of threads
    }
    #endif
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

#endif  // MAGMA_HAVE_CUDA

#ifdef __cplusplus
} // extern "C"
#endif
