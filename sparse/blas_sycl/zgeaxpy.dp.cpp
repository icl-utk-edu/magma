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

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
void 
zgeaxpy_kernel( 
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dx, 
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            int idx = row + j*num_rows;
            dy[ idx ] = alpha * dx[ idx ] + beta * dy[ idx ];
        }
    }
}

/**
    Purpose
    -------
    
    This routine computes Y = alpha *  X + beta * Y on the GPU.
    The input format is magma_z_matrix. It can handle both,
    dense matrix (vector block) and CSR matrices. For the latter,
    it interfaces the cuSPARSE library.
    
    Arguments
    ---------

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier.
                
    @param[in]
    X           magma_z_matrix
                input/output matrix Y.
                
    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier.
                
    @param[in,out]
    Y           magma_z_matrix*
                input matrix X.
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_zgeaxpy(
    magmaDoubleComplex alpha,
    magma_z_matrix X,
    magmaDoubleComplex beta,
    magma_z_matrix *Y,
    magma_queue_t queue )
{
    int m = X.num_rows;
    int n = X.num_cols;
    magma_z_matrix C={Magma_CSR};
    
    if( X.storage_type == Magma_DENSE && Y->storage_type == Magma_DENSE ){

        sycl::range<3> grid(1, 1, magma_ceildiv(m, BLOCK_SIZE));
        magma_int_t threads = BLOCK_SIZE;
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto Y_dval_ct5 = Y->dval;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgeaxpy_kernel(m, n, alpha, X.dval, beta, Y_dval_ct5,
                                       item_ct1);
                    });
            });

    } else if( X.storage_type == Magma_CSR && Y->storage_type == Magma_CSR ) {
        
        magma_zcuspaxpy( &alpha, X, &beta, *Y, &C, queue );
        magma_zmfree( Y, queue );
        magma_zmtransfer( C, Y, Magma_DEV, Magma_DEV, queue );
        magma_zmfree( &C, queue );
    } else {
        printf("%% error: matrix addition only supported for DENSE and CSR format.\n");   
    }
                    
    return MAGMA_SUCCESS;
}
