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
#define BLOCK_SIZE1 32
#define GRID_SIZE1 32768
#define GRID_SIZE22 1024
#define GRID_SIZE2 1024
#define GRID_SIZE3 32
#define GRID_SIZE4 1
#define THRS_PER_THREAD 8

#define GRID_FOR_VOLTA 160
#define REDUCE_FOR_VOLTA 5
#define BLOCK_FOR_VOLTA 32



// kernel for finding the largest element
__global__ void 
magma_zfindlargest_kernel( 
    magma_int_t total_size,
    magmaDoubleComplex *val,
    float *float_val,
    float * max )
{
    magma_int_t tidx = threadIdx.x;   
    magma_int_t bidx = blockIdx.x;
    magma_int_t gtidx = bidx * blockDim.x + tidx;
    
    magma_int_t dim_grid = 32768*32;
    
    
    magma_int_t steps = magma_ceildiv( total_size, dim_grid );
   
    float sval = 0.0;
    float lval = 0.0;
    for (magma_int_t z=0; z<steps; z++) {
        magma_int_t el = z*dim_grid + gtidx;
        sval = (float)MAGMA_Z_ABS(val[(el)%total_size]);
        float_val[(el)%total_size] = sval;
        lval = (sval > lval) ? sval : lval;
    }
    float maxval=0.0;
    
    // now reduce among threads of the warp
    #if __CUDA_ARCH__ >= 300
    #if __CUDACC_VER_MAJOR__ < 9
        #pragma unroll
        for (int z=0; z<32; z++) {
            sval = __shfl( sval,(tidx+1)%32);
            maxval = sval > maxval ? sval : maxval ;
        }
    #else
        #pragma unroll
        for (int z=0; z<32; z++) {
            sval = __shfl_sync(0xffffffff,sval, (tidx+1)%32);
            maxval = sval > maxval ? sval : maxval ;
        }
    #endif
    #endif
    
    max[bidx] = maxval;
}


// kernel for counting elements for different thresholds
__global__ void 
zthreshselect_kernel( 
    magma_int_t sampling,
    magma_int_t total_size,
    magma_int_t subset_size_g,
    float *val,
    float scaling,
    float * thrs,
    magmaDoubleComplex *dummy) // dummy argument to avoid symbol duplication for z c d s
{
    magma_int_t tidx = threadIdx.x;   
    magma_int_t bidx = blockIdx.x;
    magma_int_t gtidx = bidx * blockDim.x + tidx;
    magma_int_t total_thrs_count = blockDim.x*gridDim.x * THRS_PER_THREAD;
   
    magma_int_t subset_size = subset_size_g / sampling;

   // now define the threshold
    float thrs_inc = (float) 1 / (float) total_thrs_count;
    __shared__ float sval[BLOCK_SIZE];
    
    // local counters
    magma_int_t count[THRS_PER_THREAD];
    #pragma unroll
    for (int t=0; t<THRS_PER_THREAD; t++) {
        count[t] = 0;
    }
    for (magma_int_t z=0; z<total_size-BLOCK_SIZE; z+=BLOCK_SIZE*sampling) {
        sval[tidx] = val[ (z+tidx)%total_size ];
        #if __CUDA_ARCH__ >= 300
        #if __CUDACC_VER_MAJOR__ < 9
            for (int k=0; k<BLOCK_SIZE; k++) {
                for (int t=0; t<THRS_PER_THREAD; t++) {
                    count[t] = (sval[k] < (gtidx*THRS_PER_THREAD+t)*thrs_inc*scaling) ?
                                                    count[t]+1 : count[t];
                }
            }       
        #else
            //#pragma unroll
            for (int k=0; k<BLOCK_SIZE; k++) {
                for (int t=0; t<THRS_PER_THREAD; t++) {
                    count[t] = (sval[k] < (gtidx*THRS_PER_THREAD+t)*thrs_inc*scaling) ? 
                                                    count[t]+1 : count[t];
                }
            }
        #endif
        #endif
        // threads that have their lowest count above the subset size return
        if (__all(count[0]>subset_size)) { 
            if ( tidx == 0 ){
                thrs[bidx] = 0.0;
            }
            return;
        }
    }
    
    // check for the largest threshold of the thread
    float maxval = 0.0;
    
    // thresholds are increasing!
    #pragma unroll
    for (int t=0; t<THRS_PER_THREAD; t++) {
        maxval = (count[t] < subset_size) ? 
            (gtidx*THRS_PER_THREAD+t)*thrs_inc*scaling : maxval ;
    }
    thrs[gtidx]=maxval;
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
    sampling    magma_int_t
                determines how many elements are considered (approximate method)
                
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
    magma_int_t sampling,
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 block1(BLOCK_SIZE1, 1, 1);
    
    dim3 grid1(GRID_SIZE1, 1, 1 );
    dim3 grid2(GRID_SIZE2, 1, 1 );
    dim3 grid3(GRID_SIZE3, 1, 1 );
    dim3 grid4(GRID_SIZE4, 1, 1 );
    dim3 grid22(GRID_SIZE22, 1, 1 );
    
    dim3 gridvolta(GRID_FOR_VOLTA, 1, 1 );
    dim3 gridvolta2(REDUCE_FOR_VOLTA, 1, 1 );
    
    dim3 grid(magma_ceildiv( total_size, BLOCK_SIZE ), 1, 1 );

    float *thrs1 = NULL, *thrs2 = NULL, *thrstmp = NULL, *float_val = NULL; 
    real_Double_t start, end;
    magmaDoubleComplex *dummy = NULL;
    
    start = magma_sync_wtime( queue );
    CHECK(magma_smalloc_cpu(&thrstmp, 1));
    CHECK(magma_smalloc(&thrs1, GRID_SIZE1));
    CHECK(magma_smalloc(&thrs2, GRID_SIZE2));
    
    CHECK(magma_smalloc(&float_val, total_size));
    end = magma_sync_wtime( queue );
    printf("\n%%allocate1: %.4e\n", end-start);
    
    
    start = magma_sync_wtime( queue );
    // add an initial setp that finds the largest element
    // go over value array, each threads finds a first "largest" element 
    // and writes to thrs1. Then do reduction to find the largest value overall.
    // start = magma_sync_wtime( queue );
    magma_zfindlargest_kernel<<<grid1, block1, 0, queue->cuda_stream()>>>
            (total_size, val, float_val, thrs1);
    magma_zreduce_thrs<<<grid2, block1, 0, queue->cuda_stream()>>>
        ( thrs1, thrs2 );
    magma_zreduce_thrs<<<grid3, block1, 0, queue->cuda_stream()>>>
        ( thrs2, thrs1 );
    magma_zreduce_thrs<<<grid4, block1, 0, queue->cuda_stream()>>>
         ( thrs1, thrs2 );
         
// magma_zprint_gpu( total_size, 1, val, total_size, queue );
 // magma_dprint_gpu( total_size, 1, ddouble_val, total_size, queue );
    end = magma_sync_wtime( queue );
    printf("%%find largest: %.4e\n", end-start);
         
    
    start = magma_sync_wtime( queue );
    magma_sgetvector(1, thrs2, 1, thrstmp, 1, queue );
    thrs[0] = (double)thrstmp[0];
    // set array to 0
    CHECK(magma_svalinit_gpu(GRID_SIZE1, thrs1, queue));
    CHECK(magma_svalinit_gpu(GRID_SIZE2, thrs2, queue));
    end = magma_sync_wtime( queue );
    printf("%%allocate2: %.4e\n", end-start);
    // now start the thresholding
    // first kernel checks how many elements are smaller than the threshold
    start = magma_sync_wtime( queue );
    zthreshselect_kernel<<<grid22, block, 0, queue->cuda_stream()>>>
        (sampling, total_size, subset_size, float_val, thrs[0], thrs1, dummy);
    // second kernel identifies the largest of these thresholds
    magma_zreduce_thrs<<<grid2, block1, 0, queue->cuda_stream()>>>
        ( thrs1, thrs2 );
    magma_zreduce_thrs<<<grid3, block1, 0, queue->cuda_stream()>>>
        ( thrs2, thrs1 );
    magma_zreduce_thrs<<<grid4, block1, 0, queue->cuda_stream()>>>
         ( thrs1, thrs2 );
         
    end = magma_sync_wtime( queue );
    printf("%%threshold seletion using standard grid: %.4e\n", end-start);
    
    
    // start = magma_sync_wtime( queue );
    // zthreshselect_kernel<<<gridvolta, block, 0, queue->cuda_stream()>>>
    //     (sampling, total_size, subset_size, float_val, thrs[0], thrs1, dummy);
    // // second kernel identifies the largest of these thresholds
    // magma_zreduce_thrs<<<gridvolta2, block1, 0, queue->cuda_stream()>>>
    //     ( thrs1, thrs2 );
    // magma_zreduce_thrs<<<grid4, block1, 0, queue->cuda_stream()>>>
    //      ( thrs2, thrs1 );
    //      
    // end = magma_sync_wtime( queue );
    // printf("threshold seletion using VOLTA grid: %.4e\n", end-start);
    
    magma_sgetvector(1, thrs2, 1, thrstmp, 1, queue );
    thrs[0] = (double)thrstmp[0];
    
cleanup:
    magma_free(float_val);
    magma_free(thrs1);
    magma_free(thrs2);
    magma_free_cpu(thrstmp);

    return info;
}
