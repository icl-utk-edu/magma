/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

*/
#include "magma_internal.h"

#define PRECISION_s

#include "gemm_template_device_defs.cuh"
#include "gesvj_kernels.cuh"

// BLK_M, BLK_N, BLK_K, DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG
// Constraints:
// -- BLK_M, BLK_N, and BLK_K fully divisible by { DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG }
// -- BLK_N >= NB
// -- (DIM_X * DIM_Y) == (DIM_Y * DIM_XU) == (DIM_XG * DIM_YG)

#ifdef MAGMA_HAVE_CUDA
// Tuning for CUDA is done only for nb = {16,32} on a GH200 system
// TODO: expand tuning for other NB values
#define sgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define sgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_16 64, 16,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_32 64, 32, 16, 16,  8, 16,  8, 16,  8

#else
// Tuning for HIP is done only for nb = {16,32} on a MI300A system
// TODO: expand tuning for other NB values
#define sgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define sgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_16 32, 16, 16, 16,  8, 16,  8, 16,  8
#define sgesvj_update_nb_32 32, 32, 16, 16,  8, 16,  8, 16,  8

#endif

////////////////////////////////////////////////////////////////////////////////
// See description of gesvj_update_vectors_template_device
// under magmablas/gesvj_kernels.cuh
extern "C"
magma_int_t
magma_sgesvj_batched_update_vectors(
    magma_int_t m, magma_int_t nb,
    float **dU0array, magma_int_t lddu0,
    float **dU1array, magma_int_t lddu1,
    float **dGarray,  magma_int_t lddg,
    magma_int_t *heevj_info, int *heevj_nsweeps,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nbx2 = 2 * nb;
    magma_int_t info = 0;
    if(m < 0) {
        info = -1;
    }
    else if(nb < 0) {
        info = -2;
    }
    else if(lddu0 < m) {
        info = -4;
    }
    else if(lddu1 < m) {
        info = -6;
    }
    else if(lddg < nbx2) {
        info = -8;
    }
    else if(batchCount < 0) {
        info = -9;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    if ( m == 0 || nb == 0 || batchCount == 0 )
        return info;

    if(nb <= 4) {
        info = gesvj_update_vectors_template_batched_kernel_driver<float, sgesvj_update_nb_4>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 8) {
        info = gesvj_update_vectors_template_batched_kernel_driver<float, sgesvj_update_nb_8>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 16) {
        info = gesvj_update_vectors_template_batched_kernel_driver<float, sgesvj_update_nb_16>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 32) {
        info = gesvj_update_vectors_template_batched_kernel_driver<float, sgesvj_update_nb_32>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else {
        info = -100;
    }

    return info;
}
