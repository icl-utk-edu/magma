/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


// every multiprocessor handles one BCSR-block to copy from A
void 
zbcsrvalcpy_kernel( 
    int size_b,
    magma_int_t num_blocks,
    magmaDoubleComplex **Aval, 
    magmaDoubleComplex **Bval ,
    sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_group(2) * 65535 + item_ct1.get_group(1) < num_blocks) {
        magmaDoubleComplex *dA =
            Aval[item_ct1.get_group(2) * 65535 + item_ct1.get_group(1)];
        magmaDoubleComplex *dB =
            Bval[item_ct1.get_group(2) * 65535 + item_ct1.get_group(1)];
        int i = item_ct1.get_local_id(2);

        while( i<size_b*size_b ){
                dB[i] = dA[i];
                i+=BLOCK_SIZE;
        }
    }
}

// every multiprocessor handles one BCSR-block to initialize with 0
void 
zbcsrvalzro_kernel( 
    int size_b,
    magma_int_t num_blocks,
    magmaDoubleComplex **Bval ,
    sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_group(2) * 65535 + item_ct1.get_group(1) < num_blocks) {
        magmaDoubleComplex *dB =
            Bval[item_ct1.get_group(2) * 65535 + item_ct1.get_group(1)];
        int i = item_ct1.get_local_id(2);
        //dB += i;

        while( i<size_b*size_b ){
                dB[i] = MAGMA_Z_ZERO;
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
    sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);

        // the grids are adapted to the number of nonzero/zero blocks 
        // the upper block-number the kernels can handle is 65535*65535
        int dimgrid1 = 65535;
        int dimgrid2 = magma_ceildiv( num_blocks, 65535 );
        int dimgrid3 = magma_ceildiv( num_zblocks, 65535 );
        sycl::range<3> dimGrid(1, dimgrid1, dimgrid2);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                           zbcsrvalcpy_kernel(size_b, num_blocks, Aval, Bval,
                                              item_ct1);
                       });

        sycl::range<3> dimGrid2(1, dimgrid1, dimgrid3);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(dimGrid2 * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                           zbcsrvalzro_kernel(size_b, num_zblocks, Bval2,
                                              item_ct1);
                       });

        return MAGMA_SUCCESS;
}
