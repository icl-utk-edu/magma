/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#ifndef MAGMA_HAVE_HIP
#include"./gbtrf_tuning/dgbtrf_batch_a100.h"
#else
#include"./gbtrf_tuning/dgbtrf_batch_mi250x.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


// =============================================================================
/// @addtogroup magma_tuning
/// @{
////////////////////////////////////////////////////////////////////////////////
// auxiliary function to determine the parameters of batch zgbtrf
void
magma_get_zgbtrf_batched_params(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magma_int_t *nb, magma_int_t *threads)
{
    // get index for kl, ku based on the rounded-up even
    // values of the input bandwidths
    int ikl = (kl + 1) / 2;
    int iku = (ku + 1) / 2;
    ikl--;
    iku--;

    // make sure kl and ku are in the range [0, 16]
    ikl = min( max(ikl, 0), 15 );
    iku = min( max(iku, 0), 15 );

    // the tuning data for nb and threads are stored in 2D
    // array of size 16x16. Each entry is for the even values
    // of (kl, ku) in the range [2, 4, 6, ..., 32]
    // TODO: generate tuning data for z-precision
    #ifndef MAGMA_HAVE_HIP
    *nb      = dgbtrf_batch_nb_a100[ikl][iku];
    *threads = dgbtrf_batch_th_a100[ikl][iku];
    #else
    *nb      = dgbtrf_batch_nb_mi250x[ikl][iku];
    *threads = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif
}

////////////////////////////////////////////////////////////////////////////////
// auxiliary function to determine the parameters of batch cgbtrf
void
magma_get_cgbtrf_batched_params(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magma_int_t *nb, magma_int_t *threads)
{
    // get index for kl, ku based on the rounded-up even
    // values of the input bandwidths
    int ikl = (kl + 1) / 2;
    int iku = (ku + 1) / 2;
    ikl--;
    iku--;

    // make sure kl and ku are in the range [0, 16]
    ikl = min( max(ikl, 0), 15 );
    iku = min( max(iku, 0), 15 );

    // the tuning data for nb and threads are stored in 2D
    // array of size 16x16. Each entry is for the even values
    // of (kl, ku) in the range [2, 4, 6, ..., 32]
    // TODO: generate tuning data for c-precision
    #ifndef MAGMA_HAVE_HIP
    *nb      = dgbtrf_batch_nb_a100[ikl][iku];
    *threads = dgbtrf_batch_th_a100[ikl][iku];
    #else
    *nb      = dgbtrf_batch_nb_mi250x[ikl][iku];
    *threads = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif
}

////////////////////////////////////////////////////////////////////////////////
// auxiliary function to determine the parameters of batch dgbtrf
void
magma_get_dgbtrf_batched_params(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magma_int_t *nb, magma_int_t *threads)
{
    // get index for kl, ku based on the rounded-up even
    // values of the input bandwidths
    int ikl = (kl + 1) / 2;
    int iku = (ku + 1) / 2;
    ikl--;
    iku--;

    // make sure kl and ku are in the range [0, 16]
    ikl = min( max(ikl, 0), 15 );
    iku = min( max(iku, 0), 15 );

    // the tuning data for nb and threads are stored in 2D
    // array of size 16x16. Each entry is for the even values
    // of (kl, ku) in the range [2, 4, 6, ..., 32]
    #ifndef MAGMA_HAVE_HIP
    *nb      = dgbtrf_batch_nb_a100[ikl][iku];
    *threads = dgbtrf_batch_th_a100[ikl][iku];
    #else
    *nb      = dgbtrf_batch_nb_mi250x[ikl][iku];
    *threads = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif
}

////////////////////////////////////////////////////////////////////////////////
// auxiliary function to determine the parameters of batch sgbtrf
void
magma_get_sgbtrf_batched_params(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magma_int_t *nb, magma_int_t *threads)
{
    // get index for kl, ku based on the rounded-up even
    // values of the input bandwidths
    int ikl = (kl + 1) / 2;
    int iku = (ku + 1) / 2;
    ikl--;
    iku--;

    // make sure kl and ku are in the range [0, 16]
    ikl = min( max(ikl, 0), 15 );
    iku = min( max(iku, 0), 15 );

    // the tuning data for nb and threads are stored in 2D
    // array of size 16x16. Each entry is for the even values
    // of (kl, ku) in the range [2, 4, 6, ..., 32]
    // TODO: generate tuning data for s-precision
    #ifndef MAGMA_HAVE_HIP
    *nb      = dgbtrf_batch_nb_a100[ikl][iku];
    *threads = dgbtrf_batch_th_a100[ikl][iku];
    #else
    *nb      = dgbtrf_batch_nb_mi250x[ikl][iku];
    *threads = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif
}

#ifdef __cplusplus
} // extern "C"
#endif
