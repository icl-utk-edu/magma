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

#define BLOCK_SIZE  512



void
magma_zlobpcg_maxpy_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    magmaDoubleComplex * X, 
    magmaDoubleComplex * Y,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2); // global row index

    if ( row < num_rows ) {
        for( int i=0; i < num_vecs; i++ ) {
            Y[ row + i*num_rows ] += X[ row + i*num_rows ];
        }
    }
}


/**
    Purpose
    -------
    
    This routine computes a axpy for a mxn matrix:
        
        Y = X + Y
        
    It replaces:
            magma_zaxpy(m*n, c_one, Y, 1, X, 1);


        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    X = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows

    @param[in]
    num_vecs    magma_int_t
                number of vectors

    @param[in]
    X           magmaDoubleComplex_ptr 
                input vector X

    @param[in,out]
    Y           magmaDoubleComplex_ptr 
                input/output vector Y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zlobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr Y,
    magma_queue_t queue )
{
    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;
     magma_int_t threads = BLOCK_SIZE;
    sycl::range<3> block(1, 1, block_size);
    sycl::range<3> grid(1, 1, magma_ceildiv(num_rows, block_size));

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zlobpcg_maxpy_kernel(num_rows, num_vecs, X, Y,
                                                      item_ct1);
                       });

    return MAGMA_SUCCESS;
}
