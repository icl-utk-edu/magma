/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

#define GBTRS_SWAP_THREADS (128)

#define GBTRS_GERU_THREADS_X (32)
#define GBTRS_GERU_THREADS_Y (4)

////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_SWAP_THREADS)
void zgbtrs_swap_kernel_batched(
        int n,
        magmaDoubleComplex** dA_array, int ldda,
        magma_int_t** dipiv_array, int j)
{
    const int ntx     = blockDim.x;
    const int tx      = threadIdx.x;
    const int batchid = blockIdx.x;

    magmaDoubleComplex* dA    = dA_array[batchid];
    magma_int_t*        dipiv = dipiv_array[batchid];

    int jp = dipiv[j] - 1; // undo fortran indexing
    if( j != jp ) {
        for(int i = tx; i < n; i+=ntx) {
            magmaDoubleComplex tmp = dA[i * ldda +  j];
            dA[i * ldda +  j]      = dA[i * ldda + jp];
            dA[i * ldda + jp]      = tmp;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_GERU_THREADS_X*GBTRS_GERU_THREADS_Y)
void zgeru_kernel_batched(
        int m, int n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, int xi, int xj, int lddx, int incx,
        magmaDoubleComplex** dY_array, int yi, int yj, int lddy, int incy,
        magmaDoubleComplex** dA_array, int ai, int aj, int ldda )
{
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int ntx     = blockDim.x;
    const int nty     = blockDim.y;
    const int gtx     = blockIdx.x * ntx + tx;
    const int batchid = blockIdx.z;

    magmaDoubleComplex* dX    = dX_array[batchid] + xj * lddx + xi;
    magmaDoubleComplex* dY    = dY_array[batchid] + yj * lddy + yi;
    magmaDoubleComplex* dA    = dA_array[batchid] + aj * ldda + ai;

    if(gtx < m) {
        for(int j = ty; j < n; j += nty) {
            dA[j * ldda + gtx] += alpha * dX[gtx * incx] * dY[j * incy];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_swap_batched(
        magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda,
        magma_int_t** dipiv_array, magma_int_t j,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nthreads = min(n, GBTRS_SWAP_THREADS);
    magma_int_t nblocks  = batchCount;

    dim3 grid(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    zgbtrs_swap_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, dA_array, ldda, dipiv_array, j);
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgeru_batched_core(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, magma_int_t xi, magma_int_t xj, magma_int_t lddx, magma_int_t incx,
        magmaDoubleComplex** dY_array, magma_int_t yi, magma_int_t yj, magma_int_t lddy, magma_int_t incy,
        magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t ntx     = min(m, GBTRS_GERU_THREADS_X);
    magma_int_t nty     = min(n, GBTRS_GERU_THREADS_Y);
    magma_int_t nblocks = magma_ceildiv(m, GBTRS_GERU_THREADS_X);

    dim3 threads(ntx, nty, 1);

    magma_int_t max_batchCount = queue->get_maxBatch();
    for(magma_int_t ib = 0; ib < batchCount; ib += max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount - ib);
        dim3 grid(nblocks, 1, ibatch);

        zgeru_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
        (m, n, alpha, dX_array + ib, xi, xj, lddx, incx, dY_array + ib, yi, yj, lddy, incy, dA_array + ib, ai, aj, ldda);
    }
}



