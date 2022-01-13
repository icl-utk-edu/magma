/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_KERNEL_BATCHED_CUH
#define TRMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "trmm_template_device.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
void trmm_template_vbatched_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n)
{
    const int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_n, NB) ) return;

    trmm_small_template_device_lNx<T, NB>(
            uplo, diag,
            my_m, my_n,
            alpha, Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
                   Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
void trmm_template_vbatched_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n)
{
    const int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_n, NB) ) return;
    trmm_small_template_device_lTx<T, NB, CONJA>(
            uplo, diag,
            my_m, my_n,
            alpha, Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
                   Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
void trmm_template_vbatched_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n)
{
    const int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_n < Ai || my_n < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_n -= max( Bj, max( Ai, Aj ) );
    my_m -= Bi;

    // check if the user forces values for m, n, and k
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NB) ) return;
    trmm_small_template_device_rNx<T, NB>(
            uplo, diag,
            my_m, my_n,
            alpha, Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
                   Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
void trmm_template_vbatched_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n)
{
    const int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_n < Ai || my_n < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    // compute the maximum allowed value for m, n based on the input offsets
    my_n -= max( Bj, max( Ai, Aj ) );
    my_m -= Bi;

    // check if the user forces values for m, n, and k
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NB) ) return;
    trmm_small_template_device_rTx<T, NB, CONJA>(
            uplo, diag,
            my_m, my_n,
            alpha, Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
                   Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_vbatched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( max_n, NB ), 1, ibatch );
        trmm_template_vbatched_lNx_kernel<T, NB><<< grid, threads, 0, queue->cuda_stream() >>>
        (uplo, diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_vbatched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( max_n, NB ), 1, ibatch );
        trmm_template_vbatched_lTx_kernel<T, NB, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
        (uplo, diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_vbatched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( max_m, NB ), 1, ibatch );
        trmm_template_vbatched_rNx_kernel<T, NB><<< grid, threads, 0, queue->cuda_stream() >>>
        (uplo, diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_vbatched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( max_m, NB ), 1, ibatch );
        trmm_template_vbatched_rTx_kernel<T, NB, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
        (uplo, diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
    }
}
#endif //TRMM_TEMPLATE_KERNEL_BATCHED_CUH
