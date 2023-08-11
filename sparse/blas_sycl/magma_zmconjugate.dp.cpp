/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


void 
magma_zmconjugate_kernel(  
    int num_rows,
    magma_index_t *rowptr, 
    magmaDoubleComplex *values ,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(row < num_rows ){
        for( int i = rowptr[row]; i < rowptr[row+1]; i++){
            values[i] = MAGMA_Z_CONJ( values[i] );
        }
    }
}



/**
    Purpose
    -------

    This function conjugates a matrix. For a real matrix, no value is changed.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmconjugate(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    sycl::range<3> grid(1, 1, magma_ceildiv(A->num_rows, BLOCK_SIZE));
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct1 = A->drow;
        auto A_dval_ct2 = A->dval;

        cgh.parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, BLOCK_SIZE),
                              sycl::range<3>(1, 1, BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                magma_zmconjugate_kernel(A_num_rows_ct0, A_drow_ct1, A_dval_ct2,
                                         item_ct1);
            });
    });

    return info;
}
