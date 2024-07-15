/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSM_TEMPLATE_SHFL_KERNEL_BATCHED_CUH
#define TRSM_TEMPLATE_SHFL_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "trsm_template_device.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_lNL_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
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
    if( blockIdx.x >= magma_ceildiv(my_n, NRHS) ) return;

    trsm_template_device_lNL<T, NB, NRHS>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_lNU_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
{
    int batchid = blockIdx.z;
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
    if( blockIdx.x >= magma_ceildiv(my_n, NRHS) ) return;

    trsm_template_device_lNU<T, NB, NRHS>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + (int)ldda[batchid] * Aj  + Ai, (int)ldda[batchid],
                   Barray[batchid] + (int)lddb[batchid] * Bj  + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_lTL_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
{
    int batchid = blockIdx.z;
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
    if( blockIdx.x >= magma_ceildiv(my_n, NRHS) ) return;

    trsm_template_device_lTL<T, NB, NRHS, CONJA>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_lTU_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
{
    int batchid = blockIdx.z;
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
    if( blockIdx.x >= magma_ceildiv(my_n, NRHS) ) return;

    trsm_template_device_lTU<T, NB, NRHS, CONJA>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_rNL_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
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

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NRHS) ) return;

    trsm_template_device_rNL<T, NB, NRHS>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_rNU_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
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

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NRHS) ) return;

    trsm_template_device_rNU<T, NB, NRHS>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_rTL_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
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

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NRHS) ) return;

    trsm_template_device_rTL<T, NB, NRHS, CONJA>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static __global__
__launch_bounds__(NRHS)
void trsm_template_vbatched_rTU_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n )
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

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if( blockIdx.x >= magma_ceildiv(my_m, NRHS) ) return;

    trsm_template_device_rTU<T, NB, NRHS, CONJA>(
            diag, my_m, my_n,
            alpha, Aarray[batchid] + Aj * (int)ldda[batchid] + Ai, (int)ldda[batchid],
                   Barray[batchid] + Bj * (int)lddb[batchid] + Bi, (int)lddb[batchid]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
void trsm_template_vbatched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NRHS, 1, 1);

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_n, NRHS), 1, ibatch );

            trsm_template_vbatched_lNL_kernel<T, NB, NRHS>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_n, NRHS), 1, ibatch );

            trsm_template_vbatched_lNU_kernel<T, NB, NRHS>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
void trsm_template_vbatched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NRHS, 1, 1);

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_n, NRHS ), 1, ibatch );

            trsm_template_vbatched_lTL_kernel<T, NB, NRHS, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_n, NRHS ), 1, ibatch );

            trsm_template_vbatched_lTU_kernel<T, NB, NRHS, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
void trsm_template_vbatched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NRHS, 1, 1);

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_m, NRHS ), 1, ibatch );

            trsm_template_vbatched_rNL_kernel<T, NB, NRHS>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_m, NRHS ), 1, ibatch );

            trsm_template_vbatched_rNU_kernel<T, NB, NRHS>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
void trsm_template_vbatched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NRHS, 1, 1);

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_m, NRHS ), 1, ibatch );

            trsm_template_vbatched_rTL_kernel<T, NB, NRHS, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( magma_ceildiv( max_m, NRHS ), 1, ibatch );

            trsm_template_vbatched_rTU_kernel<T, NB, NRHS, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// template wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int NRHS>
void trsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t max_m, magma_int_t max_n,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t shape = 0;
    if      (side == MagmaLeft  && transA == MagmaNoTrans   ) { shape = 0; } // left  - NoTrans   (lNx)
    else if (side == MagmaLeft  && transA == MagmaTrans     ) { shape = 1; } // left  - Trans     (lTx)
    else if (side == MagmaLeft  && transA == MagmaConjTrans ) { shape = 2; } // left  - ConjTrans (lCx)
    else if (side == MagmaRight && transA == MagmaNoTrans   ) { shape = 3; } // right - NoTrans   (rNx)
    else if (side == MagmaRight && transA == MagmaTrans     ) { shape = 4; } // right - Trans     (rTx)
    else if (side == MagmaRight && transA == MagmaConjTrans ) { shape = 5; } // right - ConjTrans (rCx)

    switch(shape)
    {
        case 0: // lNx
            trsm_template_vbatched_lNx<T, NB, NRHS>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 1: // lTx
            trsm_template_vbatched_lTx<T, NB, NRHS, 0>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 2: // lCx
            trsm_template_vbatched_lTx<T, NB, NRHS, 1>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 3: // rNx
            trsm_template_vbatched_rNx<T, NB, NRHS>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 4: // rTx
            trsm_template_vbatched_rTx<T, NB, NRHS, 0>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 5: // rCx
            trsm_template_vbatched_rTx<T, NB, NRHS, 1>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        default:; // propose something
    }
}

#endif //TRSM_TEMPLATE_SHFL_KERNEL_BATCHED_CUH
