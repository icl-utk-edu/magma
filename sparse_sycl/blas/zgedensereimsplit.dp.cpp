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
zgedensereimsplit_kernel( 
    int num_rows, 
    int num_cols,  
    magma_index_t* rowidx,
    magmaDoubleComplex * A, 
    magmaDoubleComplex * ReA, 
    magmaDoubleComplex * ImA ,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            ReA[j] = MAGMA_Z_MAKE(std::real(A[j]), 0.0);
            ImA[j] = MAGMA_Z_MAKE(std::imag(A[j]), 0.0);
        }
    }
}

/**
    Purpose
    -------
    
    This routine takes an input matrix A in DENSE format and located on the GPU
    and splits it into two matrixes ReA and ImA containing the real and the 
    imaginary contributions of A.
    The output matrices are allocated within the routine.
    
    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A.
                
    @param[out]
    ReA         magma_z_matrix*
                output matrix contaning real contributions.
                
    @param[out]
    ImA         magma_z_matrix*
                output matrix contaning complex contributions.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_zgedensereimsplit(
    magma_z_matrix A,
    magma_z_matrix *ReA,
    magma_z_matrix *ImA,
    magma_queue_t queue )
{
    magma_zmtransfer( A, ReA, Magma_DEV, Magma_DEV, queue );
    magma_zmtransfer( A, ImA, Magma_DEV, Magma_DEV, queue );
        
    int m = A.num_rows;
    int n = A.num_cols;
    sycl::range<3> grid(1, 1, magma_ceildiv(m, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;
    /*
    DPCT1049:515: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto ReA_dval_ct4 = ReA->dval;
        auto ImA_dval_ct5 = ImA->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zgedensereimsplit_kernel(m, n, A.row, A.dval,
                                                      ReA_dval_ct4,
                                                      ImA_dval_ct5, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}
