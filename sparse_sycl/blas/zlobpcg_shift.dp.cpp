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
#include <cmath>

void
magma_zlobpcg_shift_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    magma_int_t shift, 
    magmaDoubleComplex * x ,
    sycl::nd_item<3> item_ct1)
{
    int idx = item_ct1.get_local_id(2); // thread in row
    int row = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2); // global block index

    if ( row<num_rows) {
        magmaDoubleComplex tmp = x[idx];
        /*
        DPCT1065:85: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if ( idx > shift-1 ) {
            idx-=shift;
            x[idx] = tmp;
            /*
            DPCT1065:86: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
}


/**
    Purpose
    -------
    
    For a Block-LOBPCG, the set of residuals (entries consecutive in memory)  
    shrinks and the vectors are shifted in case shift residuals drop below 
    threshold. The memory layout of x is:

        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    x = | x1[2] x2[2] x3[2] | = x1[0] x2[0] x3[0] x1[1] x2[1] x3[1] x1[2] .
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
    shift       magma_int_t
                shift number

    @param[in,out]
    x           magmaDoubleComplex_ptr 
                input/output vector x

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zlobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue )
{
    magma_int_t num_threads = num_vecs;
    // every thread handles one row containing the 
    if (  num_threads > 1024 )
        printf("error: too many threads requested.\n");

    /*
    DPCT1083:88: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = num_threads * sizeof(magmaDoubleComplex);
    if (  Ms > 1024*8 )
        printf("error: too much shared memory requested.\n");

    sycl::range<3> block(1, 1, num_threads);

    int dimgrid1 = int( sqrt( double( num_rows )));
    int dimgrid2 = magma_ceildiv( num_rows, dimgrid1 );

    sycl::range<3> grid(1, dimgrid2, dimgrid1);

    /*
    DPCT1049:87: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zlobpcg_shift_kernel(num_rows, num_vecs, shift,
                                                      x, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
