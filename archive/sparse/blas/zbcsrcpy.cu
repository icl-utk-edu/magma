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


// every multiprocessor handles one BCSR-block to copy from A
__global__ void 
zbcsrvalcpy_kernel( 
    int size_b,
    magma_int_t num_blocks,
    magmaDoubleComplex **Aval, 
    magmaDoubleComplex **Bval )
{
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        magmaDoubleComplex *dA = Aval[ blockIdx.x*65535+blockIdx.y ];
        magmaDoubleComplex *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;

        while( i<size_b*size_b ){
                dB[i] = dA[i];
                i+=BLOCK_SIZE;
        }
    }
}

// every multiprocessor handles one BCSR-block to initialize with 0
__global__ void 
zbcsrvalzro_kernel( 
    int size_b,
    magma_int_t num_blocks,
    magmaDoubleComplex **Bval )
{
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        magmaDoubleComplex *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;
        //dB += i;

        while( i<size_b*size_b ){
                dB[i] = MAGMA_Z_MAKE(0.0, 0.0);
                i+=BLOCK_SIZE;
        }
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
    size_b      magma_int_t
                blocksize in BCSR

    @param[in]
    num_blocks  magma_int_t
                number of nonzero blocks

    @param[in]
    num_zblocks magma_int_t
                number of zero-blocks (will later be filled)

    @param[in]
    Aval        magmaDoubleComplex_ptr *
                pointers to the nonzero blocks in A

    @param[in]
    Bval        magmaDoubleComplex_ptr *
                pointers to the nonzero blocks in B

    @param[in]
    Bval2       magmaDoubleComplex_ptr *
                pointers to the zero blocks in B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zblocks, 
    magmaDoubleComplex_ptr *Aval, 
    magmaDoubleComplex_ptr *Bval,
    magmaDoubleComplex_ptr *Bval2,
    magma_queue_t queue )
{
    dim3 dimBlock( BLOCK_SIZE, 1, 1 );

        // the grids are adapted to the number of nonzero/zero blocks 
        // the upper block-number the kernels can handle is 65535*65535
        int dimgrid1 = 65535;
        int dimgrid2 = magma_ceildiv( num_blocks, 65535 );
        int dimgrid3 = magma_ceildiv( num_zblocks, 65535 );
        dim3 dimGrid( dimgrid2, dimgrid1, 1 );

        zbcsrvalcpy_kernel<<< dimGrid,dimBlock, 0, queue->cuda_stream() >>>
                            ( size_b, num_blocks, Aval, Bval );

        dim3 dimGrid2( dimgrid3, dimgrid1, 1 );

        zbcsrvalzro_kernel<<< dimGrid2,dimBlock, 0, queue->cuda_stream() >>>
                            ( size_b, num_zblocks, Bval2 );

        return MAGMA_SUCCESS;
}
