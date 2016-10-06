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
    return ZPOTRF_SWITCH;
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_cpotrf_batched_crossover()
{
    return CPOTRF_SWITCH;
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_dpotrf_batched_crossover()
{
    return DPOTRF_SWITCH;
}

/// @see magma_get_zpotrf_batched_crossover
magma_int_t magma_get_spotrf_batched_crossover()
{
    return SPOTRF_SWITCH;
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


// =============================================================================
/// @}
// end group magma_tuning

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
