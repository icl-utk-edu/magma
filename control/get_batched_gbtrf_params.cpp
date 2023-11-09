/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#ifdef MAGMA_HAVE_CUDA
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
    // Each entry has the tuning value for square sizes in
    // the range [32, 64, 96, ..., 1024]
    // TODO: generate tuning data for d precision
    #ifdef MAGMA_HAVE_CUDA
    const int* nb_record = dgbtrf_batch_nb_a100[ikl][iku];
    const int* th_record = dgbtrf_batch_th_a100[ikl][iku];
    #else
    const int* nb_record = dgbtrf_batch_nb_mi250x[ikl][iku];
    const int* th_record = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif

    const int minmn = min(m, n);
    int isize = (minmn + 32 - 1) / 32;
    isize --;
    isize = min( max(isize, 0), 31 );

    *nb = nb_record[isize];
    *threads = th_record[isize];
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
    // Each entry has the tuning value for square sizes in
    // the range [32, 64, 96, ..., 1024]
    // TODO: generate tuning data for d precision
    #ifdef MAGMA_HAVE_CUDA
    const int* nb_record = dgbtrf_batch_nb_a100[ikl][iku];
    const int* th_record = dgbtrf_batch_th_a100[ikl][iku];
    #else
    const int* nb_record = dgbtrf_batch_nb_mi250x[ikl][iku];
    const int* th_record = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif

    const int minmn = min(m, n);
    int isize = (minmn + 32 - 1) / 32;
    isize --;
    isize = min( max(isize, 0), 31 );

    *nb = nb_record[isize];
    *threads = th_record[isize];
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
    // Each entry has the tuning value for square sizes in
    // the range [32, 64, 96, ..., 1024]
    // TODO: generate tuning data for d precision
    #ifdef MAGMA_HAVE_CUDA
    const int* nb_record = dgbtrf_batch_nb_a100[ikl][iku];
    const int* th_record = dgbtrf_batch_th_a100[ikl][iku];
    #else
    const int* nb_record = dgbtrf_batch_nb_mi250x[ikl][iku];
    const int* th_record = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif

    const int minmn = min(m, n);
    int isize = (minmn + 32 - 1) / 32;
    isize --;
    isize = min( max(isize, 0), 31 );

    *nb = nb_record[isize];
    *threads = th_record[isize];
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
    // Each entry has the tuning value for square sizes in
    // the range [32, 64, 96, ..., 1024]
    // TODO: generate tuning data for d precision
    #ifdef MAGMA_HAVE_CUDA
    const int* nb_record = dgbtrf_batch_nb_a100[ikl][iku];
    const int* th_record = dgbtrf_batch_th_a100[ikl][iku];
    #else
    const int* nb_record = dgbtrf_batch_nb_mi250x[ikl][iku];
    const int* th_record = dgbtrf_batch_th_mi250x[ikl][iku];
    #endif

    const int minmn = min(m, n);
    int isize = (minmn + 32 - 1) / 32;
    isize --;
    isize = min( max(isize, 0), 31 );

    *nb = nb_record[isize];
    *threads = th_record[isize];
}

#ifdef __cplusplus
} // extern "C"
#endif
