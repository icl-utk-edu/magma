/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "zgetf2_devicefunc.cuh"

#define PRECISION_z

/******************************************************************************/
__global__ void
izamax_kernel_vbatched(
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        int step, magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep )
{
    extern __shared__ double sdata[];

    const int batchid = blockIdx.x;

    // compute the actual length
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < (Ai+step) || my_N < (Aj+step) ) return;

    // compute the length of the vector for each matrix
    my_M -= (Ai+step);

    // check the length
    if(my_M <= 0) return;

    magmaDoubleComplex *dA = dA_array[batchid] + (Aj+step) * my_ldda + (Ai+step);
    magma_int_t *ipiv = ipiv_array[batchid] + Ai;
    int tx = threadIdx.x;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(my_M, dA, 1, shared_x, shared_idx);

    if (tx == 0) {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[batchid] = shared_idx[0] + step + gbstep + 1;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_vbatched(
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t step, magma_int_t** ipiv_array, magma_int_t *info_array,
        magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    izamax_kernel_vbatched<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
    (M, N, dA_array, Ai, Aj, ldda, step, ipiv_array, info_array, gbstep);

    return 0;
}

/******************************************************************************/
__global__
void zswap_kernel_vbatched(
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t step, magma_int_t** ipiv_array )
{
    const int batchid = blockIdx.x;
    const int my_ldda = (int)ldda[batchid];
    int my_M          = (int)M[batchid];
    int my_N          = (int)N[batchid];

    // check if offsets produce out-of-bound pointers
    // Here, 'step' account only for my_M, not my_N
    // (step = the row that is about to be swapped with the row having the pivot)
    if( my_M < (Ai+step) || my_N < Aj ) return;

    my_N -= Aj;

    magmaDoubleComplex *dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *ipiv = ipiv_array[batchid] + Ai;

    zswap_device(my_N, dA, my_ldda, step, ipiv);
}

/******************************************************************************/
extern "C" magma_int_t
magma_zswap_vbatched(
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t step, magma_int_t** ipiv_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    zswap_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
    (M, N, dA_array, Ai, Aj, ldda, step, ipiv_array);
    return 0;
}

/******************************************************************************/
__global__
void zscal_zgeru_1d_generic_kernel_vbatched(
        magma_int_t *M, magma_int_t *N, int step,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t *info_array, int gbstep)
{
    const int batchid = blockIdx.z;
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];

    if( my_M <= (Ai+step) || my_N <= (Aj+step) ) return;
    my_M -= Ai;
    my_N -= Aj;

    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_generic_device(my_M, my_N, step, dA, my_ldda, info, gbstep);
}


/******************************************************************************/
extern "C"
magma_int_t magma_zscal_zgeru_vbatched(
        magma_int_t *M, magma_int_t *N, magma_int_t max_M, magma_int_t step,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    dim3 threads(tbx, 1, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(magma_ceildiv(max_M,tbx), 1, ibatch);

        zscal_zgeru_1d_generic_kernel_vbatched<<<grid, threads, 0, queue->cuda_stream()>>>
        (M+i, N+i, step, dA_array+i, Ai, Aj, ldda+i, info_array+i, gbstep);
    }
    return 0;
}
