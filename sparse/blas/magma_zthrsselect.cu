/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// kernel
__global__ void 
zthreshselect_kernel( 
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs )
{
    int tidx = blockIdx.x*blockDim.x+threadIdx.x;

    if(tidx<total_size){
        ;
        // do something ?
    }
}



/**
    Purpose
    -------
    
    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.
    
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
    thrs        double*  
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
    dim3 grid(magma_ceildiv(total_size, BLOCK_SIZE), 1, 1 );
    
    zthreshselect_kernel<<<grid, block, 0, queue->cuda_stream()>>>
        (total_size, subset_size, val, thrs);

    return info;
}
