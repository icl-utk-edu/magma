/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 32
#define GRID_SIZE1 32768
#define GRID_SIZE2 1024
#define GRID_SIZE3 32
#define GRID_SIZE4 1
#define THRS_PER_THREAD 32


// kernel for counting elements for different thresholds
__global__ void 
zthreshselect_kernel( 
    magma_int_t total_size,
    magma_int_t subset_size,
    const magmaDoubleComplex * __restrict__ val,
    float * thrs )
{
    magma_int_t tidx = threadIdx.x;   
    magma_int_t bidx = blockIdx.x;
    magma_int_t gtidx = bidx * blockDim.x + tidx;
    magma_int_t total_thrs_count = blockDim.x*gridDim.x * THRS_PER_THREAD;
    // now define the threshold
    
    float thrs_inc = (float) 1 / (float) total_thrs_count;
    
    if ( tidx == 0 ){
            thrs[bidx] = 0.0;
    }
    
    // local counters
    magma_int_t count[THRS_PER_THREAD];
    #pragma unroll
    for (int t=0; t<THRS_PER_THREAD; t++) {
        count[t] = 0;
    }
    
    #pragma unroll
    for (magma_int_t z=0; z<total_size; z+=32) {
        float lval = (float)MAGMA_Z_ABS(val[(z+tidx)%total_size]);
        
        #if __CUDA_ARCH__ >= 300
        #if __CUDACC_VER_MAJOR__ < 9
            #pragma unroll
            for (int k=0; k<32; k++) {
                lval = __shfl( lval,(tidx+1)%32);
                #pragma unroll
                for (int t=0; t<THRS_PER_THREAD; t++) {
                    count[t] = (lval < (gtidx*THRS_PER_THREAD+t)*thrs_inc) ? 
                                                    count[t]+1 : count[t];
                }
            }
        #else
            #pragma unroll
            for (int k=0; k<32; k++) {
                lval = __shfl_sync(0xffffffff,lval, (tidx+1)%32);
                for (int t=0; t<THRS_PER_THREAD; t++) {
                    count[t] = (lval < (gtidx*THRS_PER_THREAD+t)*thrs_inc) ? 
                                                    count[t]+1 : count[t];
                }
            }
        #endif
        #endif
        // threads that have their lowest count above the subset size return
        if (count[0]>subset_size) { 
            return;
        }
    }
    
    // check for the largest threshold of the thread
    float maxval = 0.0;
    
    // thresholds are increasing!
    #pragma unroll
    for (int t=0; t<THRS_PER_THREAD; t++) {
        maxval = (count[t] < subset_size) ? 
            (gtidx*THRS_PER_THREAD+t)*thrs_inc : maxval ;
    }
    float thrs_loc = maxval;
    
    // now reduce among threads of the warp
    #if __CUDA_ARCH__ >= 300
    #if __CUDACC_VER_MAJOR__ < 9
        #pragma unroll
        for (int z=0; z<31; z++) {
            thrs_loc = __shfl( thrs_loc,(tidx+1)%32);
            maxval = thrs_loc > maxval ? thrs_loc : maxval ;
        }
    #else
        #pragma unroll
        for (int z=0; z<31; z++) {
            thrs_loc = __shfl_sync(0xffffffff,thrs_loc, (tidx+1)%32);
            maxval = thrs_loc > maxval ? thrs_loc : maxval ;
        }
    #endif
    #endif
    
    if ( tidx == 0 ){
            thrs[bidx] = maxval;
    }
}


// kernel identifying the best threshold
__global__ void
magma_zreduce_thrs( 
    float *thrs,
    float *thrs2)
{
    int tidx = threadIdx.x;   
    int bidx = blockIdx.x;
    int gtidx = bidx * blockDim.x + tidx;
    
    float val = thrs[gtidx];
    float maxval = val;
    
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
    #pragma unroll
    for (int z=0; z<31; z++) {
        val = __shfl(val, (tidx+1)%32);
        maxval = val > maxval ? val : maxval ;
    }
#else
    #pragma unroll
    for (int z=0; z<31; z++) {
        val = __shfl_sync(0xffffffff,val, (tidx+1)%32);
        maxval = val > maxval ? val : maxval ;
    }
#endif
#endif

    if ( tidx == 0 ){
            thrs2[bidx] = maxval;
    }
}



/**
    Purpose
    -------
    
    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.
    Hilarious approach: 
    Start a number of threads, each thread uses a pre-defined threshold, then
    checks for each element whether it is smaller than the threshold.
    In the end a global reduction identifies the threshold that is closest.

    Assuming all values are in (0,1), the distinct thresholds are defined as:
    
    threshold [ thread ] = thread / num_threads
    
    We obviously need to launch many threads.
    
    Arguments
    ---------
                
    @param[in]
    total_size  magma_int_t
                size of array val
                
    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate
                
    @param[in]
    val         magmaDoubleComplex
                array containing the values
                
    @param[out]
    thrs        float*  
                computed threshold

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zthrsholdselect(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    dim3 block(BLOCK_SIZE, 1, 1);
    
    dim3 grid1(GRID_SIZE1, 1, 1 );
    dim3 grid2(GRID_SIZE2, 1, 1 );
    dim3 grid3(GRID_SIZE3, 1, 1 );
    dim3 grid4(GRID_SIZE4, 1, 1 );
    
    float *thrs1, *thrs2, *thrstmp; 
    real_Double_t start, end;
    
    CHECK(magma_smalloc_cpu(&thrstmp, 1));
    CHECK(magma_smalloc(&thrs1, GRID_SIZE2));
    CHECK(magma_smalloc(&thrs2, GRID_SIZE3));
    
    //__global__ __launch_bounds__(32);
    
    printf("scan:... ");
    start = magma_sync_wtime( queue );
    // first kernel checks how many elements are smaller than the threshold
    zthreshselect_kernel<<<grid2, block, 0, queue->cuda_stream()>>>
        (total_size, subset_size, val, thrs1);
    end = magma_sync_wtime( queue );
    printf( "done in %.4e sec\n",(end-start) );
    
    printf("reduction:...");
    start = magma_sync_wtime( queue );
    // second kernel identifies the largest of these thresholds
    magma_zreduce_thrs<<<grid3, block, 0, queue->cuda_stream()>>>
        ( thrs1, thrs2 );
    magma_zreduce_thrs<<<grid4, block, 0, queue->cuda_stream()>>>
        ( thrs2, thrs1 );
    //magma_zreduce_thrs<<<grid4, block, 0, queue->cuda_stream()>>>
    //     ( thrs1, thrs2 );
    end = magma_sync_wtime( queue );
    printf( "done in %.4e sec\n",(end-start) );
    
    magma_sgetvector(1, thrs1, 1, thrstmp, 1, queue );
    thrs[0] = (double)thrstmp[0];
    
cleanup:
    magma_free(thrs1);
    magma_free(thrs2);
    magma_free_cpu(thrstmp);

    return info;
}
