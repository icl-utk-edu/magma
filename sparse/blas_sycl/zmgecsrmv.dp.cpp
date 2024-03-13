/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


void 
zmgecsrmv_kernel( 
    int num_rows, 
    int num_cols, 
    int num_vecs,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;
    auto dot = (magmaDoubleComplex *)dpct_local;

    if( row<num_rows ){
        for( int i=0; i<num_vecs; i++ )
                dot[item_ct1.get_local_id(2) +
                    i * item_ct1.get_local_range(2)] = MAGMA_Z_ZERO;
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( j=start; j<end; j++ ){
            int col = dcolind [ j ];
            magmaDoubleComplex val = dval[ j ];
            for( int i=0; i<num_vecs; i++ )
                dot[item_ct1.get_local_id(2) +
                    i * item_ct1.get_local_range(2)] +=
                    val * dx[col + i * num_cols];
        }
        for( int i=0; i<num_vecs; i++ )
            dy[row + i * num_cols] =
                alpha * dot[item_ct1.get_local_id(2) +
                            i * item_ct1.get_local_range(2)] +
                beta * dy[row + i * num_cols];
    }
}



/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is CSR. 
    
    Arguments
    ---------
    
    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 
                
    @param[in]
    num_vecs    mama_int_t
                number of vectors
    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in CSR

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of A in CSR

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in CSR

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(m, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;
    /*
    DPCT1083:232: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    unsigned int MEM_SIZE =
        num_vecs * BLOCK_SIZE * sizeof(magmaDoubleComplex); // num_vecs vectors
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(MEM_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zmgecsrmv_kernel(m, n, num_vecs, alpha, dval,
                                              drowptr, dcolind, dx, beta, dy,
                                              item_ct1,
                                              dpct_local_acc_ct1.get_pointer());
                         });
    });

    return MAGMA_SUCCESS;
}
