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
// see description of gesvj_setup_ptr_arrays_kernel
// under magmablas/gesvj_kernels.cuh
extern "C"
void magma_zgesvj_batched_setup_ptr_arrays(
        magma_vec_t jobv, magma_int_t i_gesvj_sweep, magma_int_t nb, magma_int_t nblk_col2,
        magmaDoubleComplex** dUi_array, magmaDoubleComplex** dUo_array, magma_int_t lddu,
        magmaDoubleComplex** dVi_array, magmaDoubleComplex** dVo_array, magma_int_t lddv,
        magmaDoubleComplex** dAgemm0_array, magmaDoubleComplex** dAgemm1_array,
        magmaDoubleComplex** dUjVj_input_array, magmaDoubleComplex** dUkVk_input_array, magmaDoubleComplex** dUjkVjk_output_array,
        magma_int_t flat_batchCount, magma_queue_t queue)
{
    gesvj_setup_ptr_arrays_kernel_batched<magmaDoubleComplex>(
        jobv, i_gesvj_sweep, nb, nblk_col2,
        dUi_array, dUo_array, lddu,
        dVi_array, dVo_array, lddv,
        dAgemm0_array, dAgemm1_array,
        dUjVj_input_array, dUkVk_input_array, dUjkVjk_output_array,
        flat_batchCount, queue);
}

////////////////////////////////////////////////////////////////////////////////
// see description of gesvj_finalize_svd_device
// under magmablas/gesvj_kernels.cuh
extern "C"
void magma_zgesvj_batched_finalize_values(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex const* const* dA_array, magma_int_t ldda,
        double** dSigma_array, magma_int_t batchCount, magma_queue_t queue)
{
    gesvj_finalize_svd_batched_kernel_driver<magmaDoubleComplex, double>
    (m, n, dA_array, ldda, dSigma_array, batchCount, queue);
}

////////////////////////////////////////////////////////////////////////////////
// see description of gesvj_finalize_vectors_batched_kernel_driver
// under magmablas/gesvj_kernels.cuh
extern "C"
void magma_zgesvj_batched_finalize_vectors(
        magma_vec_t jobu, magma_vec_t jobv, magma_int_t m, magma_int_t n,
        magmaDoubleComplex const* const* dUi_array, magma_int_t lddui,
        magmaDoubleComplex const* const* dVi_array, magma_int_t lddvi,
        magmaDoubleComplex            ** dUo_array, magma_int_t ldduo,
        magmaDoubleComplex            ** dVo_array, magma_int_t lddvo,
        double** dSigma_array, magma_int_t** index_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    gesvj_finalize_vectors_batched_kernel_driver<magmaDoubleComplex, double>
    ( jobu, jobv, m, n,
      dUi_array, lddui, dVi_array, lddvi,
      dUo_array, ldduo, dVo_array, lddvo,
      dSigma_array, index_array, batchCount, queue);
}


/******************************************************************************/
/*********************     GESVJ TEST CONVERGENCE        **********************/
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
// Do not use magma_int_t for dheevj_nsweeps or dheevj_mask
// these are internal arrays and always use `int` regardless of magma_int_t
// - gesvj_iters_per_sweep: number of iterations in a Jacobi sweep
// - sub_batch: number of parallel block-column pairs
// - dheevj_info: info of all Hermitian eigenvalue problems (across all iterations in an SVD sweep)
// - dheevj_nsweeps: number of sweeps of all eigenvalue problems (across all iterations in an SVD sweep)
// - dheevj_mask: output used flag to mask-off certain problems in the next sweep
// - all_converged: output flag for global convergence
__global__ void
zgesvj_test_convergence_kernel_batched(
        int gesvj_iters_per_sweep,
        int sub_batch, int batchCount,
        magma_int_t* dheevj_info, int* dheevj_nsweeps, int* dheevj_mask, int* all_converged )
{
    extern __shared__ int idata[];

    int* s_sweep_sum = (int*)idata;
    int* s_info_sum  = s_sweep_sum + blockDim.x * blockDim.y;

    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int bx    = blockIdx.x;
    const int bdimx = blockDim.x;
    const int bdimy = blockDim.y;
    const int pb_id = bx * bdimy + ty;

    const int pb_id_capped    = min(pb_id, batchCount-1);
    const int pb_count        = min(bdimy, batchCount - bx*bdimy);
    const int flat_batchCount = sub_batch * batchCount;

    // for each svd problem, we have `sub_batch * gesvj_iters_per_sweep` eigenvalue
    // problems. An SVD problem converges when all the eigenvalue problems exit with
    // zero info after one sweep.
    // so the batch svd convergence is when all #sweeps from batch heevj is all ones,
    // and all info's are zeros
    // This means that convergence sweep sum is as given below,
    // and convergence info sum is zero
    const int convergence_sweep_sum = sub_batch * gesvj_iters_per_sweep;
    const int convergence_info_sum  = 0;

    magma_int_t* pb_info    = dheevj_info    + pb_id_capped * sub_batch;
    int* pb_nsweeps = dheevj_nsweeps + pb_id_capped * sub_batch;
    int* pb_mask    = dheevj_mask    + pb_id_capped * sub_batch;

    // reduce nsweeps per problem
    int sweep_sum = 0;
    int info_sum  = 0;
    for(int iter = 0; iter < gesvj_iters_per_sweep; iter++) {
        for(int i = tx; i < sub_batch; i+=bdimx) {
            sweep_sum += pb_nsweeps[i];
            info_sum  += (int)(pb_info[i]);
        }
        pb_nsweeps += flat_batchCount;
        pb_info    += flat_batchCount;
    }

    s_sweep_sum[ty * bdimx + tx] = sweep_sum;
    s_info_sum [ty * bdimx + tx] = info_sum;
    __syncthreads();

    sweep_sum = 0;
    info_sum  = 0;
    for(int i = 0; i < bdimx; i++) {
        sweep_sum += s_sweep_sum[ty * bdimx + i];
        info_sum  += s_info_sum [ty * bdimx + i];
    }
    __syncthreads();

    magma_int_t mask  = (sweep_sum == convergence_sweep_sum && info_sum == convergence_info_sum) ? 0 : 1;
    if(tx == 0) {
        s_sweep_sum[ty] = mask;
    }
    __syncthreads();

    // write mask values
    if(pb_id < batchCount) {
        for(int i = tx; i < sub_batch; i+=bdimx) {
            pb_mask[i] = mask;
        }
    }

    // write the sum of masks
    int sum = 0;
    for(int i = 0; i < pb_count; i++) sum += s_sweep_sum[i];
    if(pb_id < batchCount && tx == 0 && ty == 0) {
        magmablas_iatomic_add(all_converged, sum);
    }
}

////////////////////////////////////////////////////////////////////////////////
// See the above description of zgesvj_test_convergence_kernel_batched
extern "C"
void magma_zgesvj_batched_test_convergence(
        magma_int_t gesvj_iters_per_sweep, magma_int_t sub_batch, magma_int_t batchCount,
        magma_int_t* dheevj_info, int* dheevj_nsweeps, int* dheevj_mask, int* all_converged, magma_queue_t queue )
{
    constexpr int max_threads_per_pb  = 32;
    constexpr int max_threads_per_blk = 128;    // must be >= max_threads_per_pb

    const int threads_per_pb = min(max_threads_per_pb, sub_batch);
    const int pbs_per_blk    = max_threads_per_blk / threads_per_pb;

    dim3 threads(threads_per_pb, pbs_per_blk, 1);
    dim3 grid(magma_ceildiv(batchCount, pbs_per_blk), 1, 1);

    size_t shmem = 2 * threads_per_pb * pbs_per_blk * sizeof(int);

    zgesvj_test_convergence_kernel_batched<<<grid, threads, shmem, queue->cuda_stream()>>>
    (gesvj_iters_per_sweep, sub_batch, batchCount, dheevj_info, dheevj_nsweeps, dheevj_mask, all_converged);
}



