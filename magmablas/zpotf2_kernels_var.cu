/*
   -- MAGMA (version 1.4) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
 */
#define PRECISION_z

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

//#define VBATCH_DISABLE_THREAD_RETURN
#ifdef VBATCH_DISABLE_THREAD_RETURN
#define ENABLE_COND1
#define ENABLE_COND2
#define ENABLE_COND4
#define ENABLE_COND5
#define ENABLE_COND6
#endif

#define MAX_NTCOL 1
#include "zpotf2_devicesfunc.cuh"
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void zpotf2_smlpout_kernel_vbatched_v2(int maxm, magma_int_t *m, 
        magmaDoubleComplex **dA_array, magma_int_t *lda, 
        int localstep, int gbstep, magma_int_t *info_array)
{
    const int batchid   = blockIdx.z;
    const int my_m      = (int)m[batchid];
    const int mylda     = (int)lda[batchid];

    const int myoff     = ((maxm - my_m)/POTF2_NB)*POTF2_NB;
    const int mylocstep = localstep - myoff;
    const int myrows    = mylocstep >= 0 ? my_m-mylocstep : 0;
    const int myib      = min(POTF2_NB, myrows);

    #ifndef VBATCH_DISABLE_THREAD_RETURN
    const int tx = threadIdx.x; 
    if(tx >=  myrows) return;
    #else
    if(myrows <= 0) return;   
    #endif
    
    if(myib == POTF2_NB)
        zpotf2_smlpout_fixwidth_device( myrows, dA_array[batchid]+mylocstep, dA_array[batchid]+mylocstep+mylocstep*mylda, mylda, mylocstep, gbstep, &(info_array[batchid]));
    else
        zpotf2_smlpout_anywidth_device( myrows, myib, dA_array[batchid]+mylocstep, dA_array[batchid]+mylocstep+mylocstep*mylda, mylda, mylocstep, gbstep, &(info_array[batchid]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void zpotf2_smlpout_kernel_vbatched(magma_int_t *m, 
        magmaDoubleComplex **dA_array, magma_int_t *lda, 
        int localstep, int gbstep, magma_int_t *info_array)
{
    const int batchid = blockIdx.z;
    const int myrows  = (int)m[batchid] - localstep;
    const int myib    = min(POTF2_NB, myrows);
    const int mylda   = lda[batchid];
    
    #ifndef VBATCH_DISABLE_THREAD_RETURN
    const int tx = threadIdx.x; 
    if(tx >=  myrows) return; 
    #else
    if(myrows <= 0) return; 
    #endif
    
    if(myib == POTF2_NB)
        zpotf2_smlpout_fixwidth_device( myrows, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*mylda, mylda, localstep, gbstep, &(info_array[batchid]));
    else
        zpotf2_smlpout_anywidth_device( myrows, myib, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*mylda, mylda, localstep, gbstep, &(info_array[batchid]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotrf_lpout_vbatched(
        magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,  
        magmaDoubleComplex **dA_array, magma_int_t *lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    // Quick return if possible
    if (max_n <= 0) {
        arginfo = -33;  // any value for now
        return arginfo;
    }

    dim3 dimGrid(1, 1, batchCount);
    for(magma_int_t j = 0; j < max_n; j+= POTF2_NB) {
        magma_int_t rows_max = max_n-j;
        magma_int_t nbth = rows_max; 
        dim3 threads(nbth, 1);
        magma_int_t shared_mem_size = sizeof(magmaDoubleComplex)*(nbth+POTF2_NB)*POTF2_NB;
        if(shared_mem_size > 47000) 
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }
        //zpotf2_smlpout_kernel_vbatched<<<dimGrid, threads, shared_mem_size, queue >>>(n, dA_array, lda, j, gbstep, info_array);
        zpotf2_smlpout_kernel_vbatched_v2
        <<<dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
        (max_n, n, dA_array, lda, j, gbstep, info_array);
    }
    return arginfo;
}
