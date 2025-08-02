/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define  blockinfo(i,j)  blockinfo[(i)*c_blocks   + (j)]
#define  val(i,j) val+((blockinfo(i,j)-1)*size_b*size_b)



// every thread initializes one entry
__global__ void 
zbcsrblockinfo5_kernel( 
    magma_int_t num_blocks,
    magmaDoubleComplex * address,
    magmaDoubleComplex **AII )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < num_blocks ){
        *AII[ i ] = *address;
        if(i==0)
        printf("address: %d\n", address);
    }
}



/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine copies the filled blocks
    from the original matrix A and initializes the blocks that will later be 
    filled in the factorization process with zeros.
    
    Arguments
    ---------


    @param[in]
    lustep      magma_int_t
                lustep

    @param[in]
    num_blocks  magma_int_t
                number of nonzero blocks

    @param[in]
    c_blocks    magma_int_t
                number of column-blocks
                
    @param[in]
    size_b      magma_int_t
                blocksize
                
    @param[in]
    blockinfo   magma_int_t*
                block filled? location?

    @param[in]
    val         magmaDoubleComplex*
                pointers to the nonzero blocks in A

    @param[in]
    AII         magmaDoubleComplex**
                pointers to the respective nonzero blocks in B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaDoubleComplex_ptr val,
    magmaDoubleComplex_ptr *AII,
    magma_queue_t queue )
{
    dim3 dimBlock( BLOCK_SIZE, 1, 1 );

    int dimgrid = magma_ceildiv( num_blocks, BLOCK_SIZE );
    dim3 dimGrid( dimgrid, 1, 1 );


    printf("dim grid: %d x %d", dimgrid, BLOCK_SIZE);
    magmaDoubleComplex **hAII;
    magma_malloc((void **)&hAII, num_blocks*sizeof(magmaDoubleComplex*));

    for(int i=0; i < num_blocks; i++) {
       hAII[i] = val(lustep,lustep);
    }
    magma_setvector( num_blocks, sizeof(magmaDoubleComplex*), 
                                                            hAII, 1, AII, 1 );
/*
    magma_setvector( 1, sizeof(magmaDoubleComplex*), address, 1, daddress, 1 );
    zbcsrblockinfo5_kernel<<< dimGrid,dimBlock, 0, queue->cuda_stream() >>>
                        ( num_blocks, daddress, AII );

*/
    return MAGMA_SUCCESS;
}
