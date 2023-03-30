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
#if   defined(MAGMA_HAVE_CUDA)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#elif defined(MAGMA_HAVE_HIP)
#include <amd_hip_cooperative_groups.h>
#endif

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
    const int nthreads = min(kl+1, GBTF2_JU_FILLIN_MAX_THREADS);
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
    const int nthreads = min(kl+ku+1, GBTF2_SWAP_MAX_THREADS);
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

/******************************************************************************/
__global__
void zgbtf2_native_kernel(
    int m, int n, int kl, int ku,
    magmaDoubleComplex *dA, int ldda, magma_int_t *ipiv,
    int* ju, int gbstep, magma_int_t *dinfo)
{
#define dA(i,j) dA[(j)*ldda + (i)]
    extern __shared__ magmaDoubleComplex zdata[];
    #ifdef MAGMA_HAVE_CUDA
    cg::grid_group grid = cg::this_grid();
    #else
    grid_group grid = this_grid();
    #endif
    const int tx     = threadIdx.x;
    const int ntx    = blockDim.x;
    const int bx     = blockIdx.x;
    const int min_mn = min(m, n);
    const int kv     = kl + ku;
    const int mband  = kv + 1 + kl;

    int linfo = (gbstep == 0) ? 0 : *dinfo;
    int swap_length = 0, local_ju = 0, jp = 0;
    double rx_abs_max = 0;
    magmaDoubleComplex tmp = MAGMA_Z_ZERO, reg = MAGMA_Z_ZERO;

    // setup shared memory
    magmaDoubleComplex* sA = zdata;
    double*             sX = (double*)(sA + mband);

    // init sA to zero & sX to [0, 1, 2, ...]
    for(int i = tx; i < mband; i+=ntx)
        sA[i] = MAGMA_Z_ZERO;
    for(int i = tx; i < (kl+1); i+=ntx)
        sX[i] = i;
    __syncthreads();

    // read column
    const int col_start  = kl + max(ku-bx,0);
    const int col_end    = kl + ku + min(kl, n-1-bx);
    //const int col_length = col_end - col_start + 1;
    for(int i = col_start+tx; i <= col_end; i+=ntx) {
        sA[i] = dA(i, bx);
    }
    __syncthreads();

    // main loop
    for(int j = 0; j < min_mn; j++) {
        int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
        // find pivot
        if(bx == j) {
            for(int i = tx; i < km; i+=ntx) {
                sX[i] = fabs( MAGMA_Z_REAL(sA[kv+i]) ) + fabs( MAGMA_Z_IMAG(sA[kv+i]) );
            }
            __syncthreads();

            rx_abs_max = sX[0];
            jp = 0;
            for(int i = 1; i < km; i++) {
                if( sX[i] > rx_abs_max ) {
                    rx_abs_max = sX[i];
                    jp         = i;
                }
            }

            linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (j+1) : linfo;

            if(tx == 0) {
                ipiv[j] = jp + j + 1;  // +1 for fortran indexing
                *dinfo   = (magma_int_t)linfo;
            }
            __threadfence();

            #ifdef DBG
            printf("[%d]: found pivot = %f @ %d -- linfo = %d\n", bx, rx_abs_max, jp, linfo);
            #endif
        }
        grid.sync();

        // read information written by j-th block
        if(bx >= j) {
            jp    = ipiv[j] - j - 1;
            linfo = (int)(*dinfo);
            #ifdef DBG
            printf("[%d]: jp = %d -- linfo = %d\n", bx, jp, linfo);
            #endif
        }
        local_ju = max(local_ju, min(gbstep+j+ku+jp, n-1));
        //swap_length = local_ju - (j+gbstep) + 1;
        __syncthreads();

        // swap
        if(bx >= j && bx <= local_ju && tx == 0) {
            if(jp != 0) {
                int j1 = (kv +  0) - (bx-j);
                int j2 = (kv + jp) - (bx-j);
                tmp    = sA[j1];
                sA[j1] = sA[j2];
                sA[j2] = tmp;
            }
        }
        __syncthreads();

        // scal & write to global memory
        if(bx == j) {
            reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sA[kv] );
            for(int i = tx; i < (km-1); i+=ntx) {
                sA[kv+1+i] *= reg;
            }
            __syncthreads();

            for(int i = tx; i < mband; i+=ntx) {
                dA(i,j) = sA[i];
            }
            __threadfence();
        }
        grid.sync();

        // ger
        if(bx > j && bx <= local_ju) {
            int j1 = (kv + 0) - (bx-j);
            for(int i = tx; i < km-1; i+=ntx) {
                #ifdef DBG
                printf("[%d]: j = %d, sA[%d] -= sA[%d] * dA(%d,%d)\n", bx, j, j1+1+i, j1, kv+1+i,j);
                #endif
                sA[j1+1+i] -= sA[j1] * dA(kv+1+i,j);
            }
            __syncthreads();
        }
    }

    // write A
    //for(int i = tx; i < mband; i+=ntx) {
    //    dA(i,bx) = sA[i];
    //}

#undef dA
}

/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_native_work(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* ipiv,
    magma_int_t* info,
    void* device_work, magma_int_t* lwork,
    magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    // calculate workspace required
    magma_int_t lwork_required = 0;
    lwork_required += 1 * sizeof(magma_int_t); // ju
    lwork_required += 1 * sizeof(magma_int_t); // dinfo

    if(*lwork < 0) {
       // query assumed
       *lwork = lwork_required;
       return *info;
    }

    if(*lwork < lwork_required) {
        *info = -11;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_int_t gbstep   = 0;
    magma_int_t nthreads = kv+1;
    magma_int_t nblocks  = n;

    // device pointers
    magma_int_t *ju    = (magma_int_t*)device_work;
    magma_int_t *dinfo = ju + 1;

    magma_int_t shmem = 0;
    shmem += mband  * sizeof(magmaDoubleComplex);
    shmem += (kl+1) * sizeof(double);

    dim3 threads(nthreads, 1, 1);
    dim3 grid(nblocks, 1, 1);


    void *kernel_args[] = {&m, &n, &kl, &ku, &dA, &ldda, &ipiv, &ju, &gbstep, &dinfo};
    cudaError_t e = cudaLaunchCooperativeKernel((void*)zgbtf2_native_kernel, grid, threads, kernel_args, shmem, queue->cuda_stream());
    magma_igetvector_async( 1, dinfo, 1, info, 1, queue );

    return *info;
}

/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_native(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* ipiv,
    magma_int_t* info, magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbtf2_native_work(m, n, kl, ku, NULL, ldda, NULL, info, NULL, lwork, queue);

    void* device_work = NULL;
    magma_malloc(&device_work, lwork[0]);

    magma_zgbtf2_native_work(m, n, kl, ku, dA, ldda, ipiv, info, device_work, lwork, queue);

    magma_free(device_work);
    return *info;
}