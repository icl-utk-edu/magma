/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Ahmad Abdelfattah

*/
#include "magma_internal.h"

#define PRECISION_z

#include "gemm_template_device_defs.cuh"
#include "gesvj_kernels.cuh"

////////////////////////////////////////////////////////////////////////////////
// See description of gesvj_update_vectors_template_device
// under magmablas/gesvj_kernels.cuh
extern "C"
magma_int_t
magma_zgesvj_batched_update_vectors(
    magma_int_t m, magma_int_t nb,
    magmaDoubleComplex **dU0array, magma_int_t lddu0,
    magmaDoubleComplex **dU1array, magma_int_t lddu1,
    magmaDoubleComplex **dGarray,  magma_int_t lddg,
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
        info = gesvj_update_vectors_template_batched_kernel_driver<magmaDoubleComplex, zgesvj_update_nb_4>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 8) {
        info = gesvj_update_vectors_template_batched_kernel_driver<magmaDoubleComplex, zgesvj_update_nb_8>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 16) {
        info = gesvj_update_vectors_template_batched_kernel_driver<magmaDoubleComplex, zgesvj_update_nb_16>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else if(nb <= 32) {
        info = gesvj_update_vectors_template_batched_kernel_driver<magmaDoubleComplex, zgesvj_update_nb_32>
        (m, nb, dU0array, lddu0, dU1array, lddu1, dGarray, lddg, heevj_info, heevj_nsweeps, batchCount, queue );
    }
    else {
        info = -100;
    }

    return info;
}
