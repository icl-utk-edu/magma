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
#include "shuffle.cuh"
#include "zgetf2_devicefunc.cuh"

#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

#define GBTF2_JU_FILLIN_MAX_THREADS (64)
#define GBTF2_SWAP_MAX_THREADS      (128)
#define GBTF2_SCAL_GER_MAX_THREADS  (64)

/******************************************************************************/
// This kernel must be called before pivot adjustment and before updating ju
__global__ __launch_bounds__(GBTF2_JU_FILLIN_MAX_THREADS)
void
zgbtf2_adjust_ju_fillin_kernel_batched(
    int n, int kl, int ku,
    magmaDoubleComplex** dAB_array, int lddab,
    magma_int_t** dipiv_array, int* ju_array, int gbstep, int batchCount)
{
    const int gtx     = blockIdx.x * blockDim.x + threadIdx.x; // global thread x-index
    const int batchid = blockIdx.z;

    //ju = max(ju, min(j+ku+jp, n-1));
    magma_int_t* ipiv = dipiv_array[batchid];
    magmaDoubleComplex *dAB = dAB_array[batchid];

    int jp   = (int)(ipiv[gbstep]) - 1;    // undo fortran indexing
    int ju1  = (gbstep == 0) ? 0 : ju_array[batchid];
    int ju2  = max(ju1, min(gbstep+ku+jp, n-1));

    if(gtx < kl) {
        for(int j = ju1+1; j <= ju2; j++) {
            dAB[j*lddab + gtx] = MAGMA_Z_ZERO;
        }
    }
}

/******************************************************************************/
// auxiliary routine that sets the necessary fill-in elements based on the new pivot
// must be called before pivot adjustment and before updating ju
extern "C"
void magma_zgbtrf_set_fillin(
        magma_int_t n, magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex** dAB_array, magma_int_t lddab,
        magma_int_t** dipiv_array, int* ju_array, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    // if kl = 0, use at least one thread to set ju
    const int nthreads = min(kl+1, (magma_int_t)GBTF2_JU_FILLIN_MAX_THREADS);
    const int nblocks  = magma_ceildiv(kl, nthreads);
    dim3 threads(nthreads, 1, 1);
    dim3 grid(nblocks, 1, batchCount);
    zgbtf2_adjust_ju_fillin_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, kl, ku, dAB_array, lddab, dipiv_array, ju_array, gbstep, batchCount);
}

/******************************************************************************/
__global__ __launch_bounds__(GBTF2_SWAP_MAX_THREADS)
void zgbtf2_swap_kernel_batched(
        magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
        magma_int_t** dipiv_array, int ipiv_offset,
        int* ju_array, magma_int_t gbstep)
{
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int batchid = blockIdx.x;
    magmaDoubleComplex *dAB = dAB_array[batchid] + aj * lddab + ai;
    magma_int_t *ipiv = dipiv_array[batchid] + ipiv_offset;

    int ju = ju_array[batchid];
    int jp = (int)ipiv[0] - 1;
    int swap_len = ju - gbstep + 1;

    if( !(jp == 0) ) {
        magmaDoubleComplex tmp;
        //magmaDoubleComplex *sR1 = &sAB(kv   ,j);
        //magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
        magmaDoubleComplex *dR1 = dAB;      // 1st row with the diagonal
        magmaDoubleComplex *dR2 = dAB + jp; // 2nd row with the pivot
        for(int i = tx; i < swap_len; i+=ntx) {
            tmp                = dR1[i * (lddab-1)];
            dR1[i * (lddab-1)] = dR2[i * (lddab-1)];
            dR2[i * (lddab-1)] = tmp;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgbtf2_zswap_batched(
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    magma_int_t** dipiv_array, magma_int_t ipiv_offset,
    int* ju_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    const int nthreads = min(kl+ku+1, (magma_int_t)GBTF2_SWAP_MAX_THREADS);
    dim3 threads(nthreads, 1, 1);
    dim3 grid(batchCount, 1, 1);

    zgbtf2_swap_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (dAB_array, ai, aj, lddab, dipiv_array, ipiv_offset, ju_array, gbstep);

    return 0;
}


/******************************************************************************/
__global__ __launch_bounds__(GBTF2_SCAL_GER_MAX_THREADS)
void zgbtf2_scal_ger_kernel_batched(
    int m, int n, int kl, int ku,
    magmaDoubleComplex **dAB_array, int ai, int aj, int lddab,
    int* ju_array, int gbstep, magma_int_t *info_array)
{
    const int gtx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int batchid = blockIdx.z;
    int ju            = ju_array[batchid];
    int swap_length   = ju - gbstep + 1;
    int km            = 1 + min( kl, m-gbstep ); // diagonal + subdiagonal(s)

    if( info_array[batchid] != 0 ) return;

    magmaDoubleComplex* dAB = dAB_array[batchid] + aj * lddab + ai;
    magmaDoubleComplex  rA  = MAGMA_Z_ZERO, reg = MAGMA_Z_ZERO;

    if( gtx > 0 && gtx < km ) {
        reg = MAGMA_Z_DIV(MAGMA_Z_ONE, dAB[0]);
        rA  = dAB[ gtx ];
        rA *= reg;
        dAB[ gtx ] = rA;

        for(int i = 1; i < swap_length; i++)
            dAB[i * (lddab-1) + gtx] -= rA * dAB[i * (lddab-1) + 0];
    }
}


/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_scal_ger_batched(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    int* ju_array, magma_int_t gbstep, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t km = 1 + min( kl, m-gbstep ); // diagonal + subdiagonal(s)
    magma_int_t nthreads = GBTF2_SCAL_GER_MAX_THREADS;
    magma_int_t nblocks  = magma_ceildiv(km, nthreads);

    dim3 threads(GBTF2_SCAL_GER_MAX_THREADS, 1, 1);

    magma_int_t max_batchCount = queue->get_maxBatch();
    for(magma_int_t s = 0; s < batchCount; s+=max_batchCount) {
        magma_int_t ibatch = min(batchCount-s, max_batchCount);
        dim3 grid(nblocks, 1, ibatch);

        zgbtf2_scal_ger_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
        (m, n, kl, ku, dAB_array+s, ai, aj, lddab, ju_array+s, gbstep, info_array+s);
    }
    return 0;
}
