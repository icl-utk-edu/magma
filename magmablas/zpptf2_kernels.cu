/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad
   @author Natalie Beams

   @precisions normal z -> s d c
 */
#include <cuda.h> // for CUDA_VERSION
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_z

#define MAX_NTCOL 8
#if defined(PRECISION_s)
#define NTCOL2   (4)
#define NTCOL1   (8)
#elif defined(PRECISION_d)
#define NTCOL2   (2)
#define NTCOL1   (4)
#else
#define NTCOL2   (1)
#define NTCOL1   (1)
#endif


#define PPTRF_MAX_NB  (8)
#define ZPPTF2_MAX_NTHREADS (1024)

#include "zpotf2_devicesfunc.cuh"
#include "zpptf2_devicesfunc.cuh"

/******************************************************************************/
template<int NB>
__global__
__launch_bounds__(ZPPTF2_MAX_NTHREADS)
void zpptf2_smlpout_fixwidth_kernel_batched(int m,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    zpptf2_smlpout_fixwidth_device<NB>
    (m, dA, localstep, dA+PACKED(localstep, localstep, lda), lda, localstep, gbstep, &(info_array[batchid]));
}


/******************************************************************************/
template<int NB>
__global__
__launch_bounds__(ZPPTF2_MAX_NTHREADS)
void zpptf2_smlpout_anywidth_kernel_batched(int m, int n,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    zpptf2_smlpout_anywidth_device<NB>
    (m, n, dA, localstep, dA+PACKED(localstep, localstep, lda), lda, localstep, gbstep, &(info_array[batchid]));
}

/******************************************************************************/
magma_int_t magma_zpptrf_lpout_get_ntcol_batched( magma_int_t m )
{
    magma_int_t ntcol = 1;
    if      (m > 64) ntcol = 1;
    else if (m > 32) ntcol = NTCOL2;
    else             ntcol = NTCOL1;
    return ntcol;
}

/******************************************************************************/
template<magma_int_t PPTF2_NB>
magma_int_t
magma_zpptrf_lpout_iteration_batched(
    magma_uplo_t uplo, magma_int_t rows, magma_int_t ib,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t local_step, magma_int_t gbstep, magma_int_t shared_mem_size,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    if ( uplo != MagmaLower ) {
        fprintf( stderr, "%s: only uplo = MagmaLower is supported\n", __func__ );
        arginfo = -1;
    } else if (rows < 0 ) {
        arginfo = -2;
    } else if ( ib < 0 ) {
        arginfo = -3;
    } else if ( ai < 0 ) {
        arginfo = -5;
    } else if ( aj < 0 ) {
        arginfo = -6;
    } else if ( lda < max(1,rows) ) {
        arginfo = -7;
    } else if ( local_step < 0 ) {
        arginfo = -8;
    } else if ( gbstep < 0 ) {
        arginfo = -9;
    } else if ( shared_mem_size < 0 ) {
        arginfo = -10;
    }else if ( batchCount < 0 ) {
        arginfo = -12;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t ntcol = magma_zpptrf_lpout_get_ntcol_batched( rows );
    dim3 dimGrid(magma_ceildiv( batchCount, ntcol ), 1, 1);
    dim3 threads(rows, ntcol);

    #if CUDA_VERSION >= 9000
    cudaFuncSetAttribute(zpptf2_smlpout_fixwidth_kernel_batched<PPTF2_NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute(zpptf2_smlpout_anywidth_kernel_batched<PPTF2_NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    #endif
    if (ib == PPTF2_NB) {
        zpptf2_smlpout_fixwidth_kernel_batched<PPTF2_NB>
        <<< dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
        (rows, dA_array, ai, aj, lda, local_step, gbstep, info_array, batchCount);
    }
    else {
        zpptf2_smlpout_anywidth_kernel_batched<PPTF2_NB>
        <<< dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
        (rows, ib, dA_array, ai, aj, lda, local_step, gbstep, info_array, batchCount);
    }

    return arginfo;
}


/******************************************************************************/
extern "C" magma_int_t
magma_zpptrf_lpout_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaLower ) {
        fprintf( stderr, "%s: only uplo = MagmaLower is supported\n", __func__ );
        arginfo = -1;
    } else if (n < 0 || n > 1024) {
        fprintf( stderr, "%s: only sizes up to 1024 are supported\n", __func__ );
        arginfo = -2;
    } else if ( ai < 0 ) {
        arginfo = -4;
    } else if ( aj < 0 ) {
        arginfo = -5;
    } else if ( lda < max(1,m) ) {
        arginfo = -6;
    } else if ( gbstep < 0 ) {
        arginfo = -7;
    } else if ( batchCount < 0 ) {
        arginfo = -9;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    const int max_nb = PPTRF_MAX_NB;
    const int min_nb = 2;

    magma_int_t  ib, rows, j = 0;
    magma_int_t shmem_max = (magma_int_t)magma_getdevice_shmem_block_optin();
    magma_int_t shared_mem_size = 0;

   // main loop
    magma_int_t pptf2_nb = -1;
    while( j < n ) {
        rows = m-j;
        magma_int_t nbth  = rows;
        magma_int_t ntcol = magma_zpptrf_lpout_get_ntcol_batched( rows );

        // choose nb
        for(int nb = max_nb; nb >= min_nb; nb >>= 1) {
            shared_mem_size = ntcol * (sizeof(magmaDoubleComplex)*(nbth+nb)*nb);
            if( shared_mem_size < shmem_max ) {
                if( pptf2_nb < 0 || (pptf2_nb > 0 && j%nb == 0) ) {
                    pptf2_nb = nb;
                }
                // else, no change to pptf2_nb
                break;
            }
        }

        if ( pptf2_nb < 0 ) {
            arginfo = -33;
            printf("Error in %s: matrix is too large, required shared memory (%.2f KB) is not supported\n", __func__, (float)(shared_mem_size)/1024.);
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        ib   = min(pptf2_nb, n-j);
        switch(pptf2_nb) {
            case  8: magma_zpptrf_lpout_iteration_batched< 8>(uplo, rows, ib, dA_array, ai, aj, lda, j, gbstep, shared_mem_size, info_array, batchCount, queue); break;
            case  4: magma_zpptrf_lpout_iteration_batched< 4>(uplo, rows, ib, dA_array, ai, aj, lda, j, gbstep, shared_mem_size, info_array, batchCount, queue); break;
            case  2: magma_zpptrf_lpout_iteration_batched< 2>(uplo, rows, ib, dA_array, ai, aj, lda, j, gbstep, shared_mem_size, info_array, batchCount, queue); break;
            default: {
                arginfo = -100;
                printf("Error in %s: unsupported blocking size %lld\n", __func__, (unsigned long long)(pptf2_nb));
                magma_xerbla( __func__, -(arginfo) );
                return arginfo;
            }
        } // end switch statement
        j += pptf2_nb;
    } // end while loop

    return arginfo;
}
