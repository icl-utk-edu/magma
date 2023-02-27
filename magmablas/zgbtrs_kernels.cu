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

#define GBTRS_UPPER_THREADS (128)

////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_SWAP_THREADS)
void zgbtrs_swap_kernel_batched(
        int k1, int k2, int n,
        magmaDoubleComplex** dA_array, int ldda,
        magma_int_t** dipiv_array)
{
    const int ntx     = blockDim.x;
    const int tx      = threadIdx.x;
    const int batchid = blockIdx.x;

    magmaDoubleComplex* dA    = dA_array[batchid];
    magma_int_t*        dipiv = dipiv_array[batchid];

    for(int j = k1; j <= k2; j++) {
        int jp = dipiv[j] - 1; // undo fortran indexing
        if( j != jp ) {
            for(int i = tx; i < n; i+=ntx) {
                magmaDoubleComplex tmp = dA[i * ldda +  j];
                dA[i * ldda +  j]      = dA[i * ldda + jp];
                dA[i * ldda + jp]      = tmp;
            }
        }

        // to make sure the writes are visible to all threads in the same block
        __syncthreads();
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
__global__
__launch_bounds__(GBTRS_UPPER_THREADS)
void zgbtrs_upper_columnwise_kernel_batched(
        int n, int kl, int ku, int nrhs, int j,
        magmaDoubleComplex** dA_array, int ldda,
        magmaDoubleComplex** dB_array, int lddb)
{
#define dA(i,j) dA[(j)*ldda + (i)]
#define dB(i,j) dB[(j)*lddb + (i)]

    const int kv      = kl + ku;
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int batchid = blockIdx.x;
    //const int je      = (n-1) - j;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dA/dB based on j
    dA += j * ldda + kv;
    dB += j;

    const int nupdates = min(kv, j);
    magmaDoubleComplex s;
    for(int rhs = 0; rhs < nrhs; rhs++) {
        s = dB(0,rhs) * MAGMA_Z_DIV(MAGMA_Z_ONE, dA(0,0));
        __syncthreads();

        if(tx == 0) dB(0,rhs) = s;
        for(int i = tx; i < nupdates ; i+= ntx) {
            dB(-i-1,rhs) -= s * dA(-i-1,0);
        }
    }

#undef dA
#undef dB
}

////////////////////////////////////////////////////////////////////////////////
/*
template<int MAX_THREADS, int NB>
__global__
__launch_bounds__(MAX_THREADS)
void zgbtrs_lower_blocked_kernel_batched(
        int n, int kl, int ku, int nrhs, int nrhs_nb,
        magmaDoubleComplex** dA_array, int ldda,
        magmaDoubleComplex** dB_array, int lddb )
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int kv      = kl + ku;
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int bx      = blockIdx.x;
    const int by      = blockIdx.y;
    const int batchid = bx;
    const int my_rhs  = min(nrhs_nb, nrhs - by * nrhs_nb);
    const int n1      = ( n / NB ) * NB;
    const int n2      = n - n1;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    magmaDoubleComplex sB     = (magmaDoubleComplex*)zdata;

    // advance dA and dB
    dA += kv;
    dB += by * nrhs_nb;

    int tmp = (n >= NB) ? NB : n2;

    if(tx < tmp) {
        for(int jb = 0; jb < my_rhs; jb++) {
            sB[jb * sldb + tx] = dB[jb * lddb + tx];
        }
    }

    for(int j = 0; j < n1; j+=NB) {
        // read A

        // read B

        // apply A

        // write part of B that is finished and shift the the rest up

        dA +=
        dB +=
    }

    // cleanup section


}
*/

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_swap_batched(
        magma_int_t k1, magma_int_t k2, magma_int_t n,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magma_int_t** dipiv_array, magma_int_t j,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nthreads = min(n, GBTRS_SWAP_THREADS);
    magma_int_t nblocks  = batchCount;

    dim3 grid(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    zgbtrs_swap_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (k1, k2, n, dA_array, ldda, dipiv_array, j);
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
    if(m == 0 || n == 0 || batchCount == 0) return;

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

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_upper_columnwise_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku,
        magma_int_t nrhs, magma_int_t j,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t kv       = kl + ku;
    magma_int_t nthreads = min(GBTRS_UPPER_THREADS, kv+1);
    magma_int_t nblocks  = batchCount;

    dim3 grid(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    zgbtrs_upper_columnwise_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, kl, ku, nrhs, j, dA_array, ldda, dB_array, lddb);
}
