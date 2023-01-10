/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// CSR-SpMV kernel
void 
zcgecsrmv_mixed_prec_kernel( 
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * ddiagval,
    magmaFloatComplex * doffdiagval,
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;

    if(row<num_rows){
        magmaDoubleComplex dot = ddiagval[ row ] * dx[ row ];
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( j=start; j<end; j++){
            magmaDoubleComplex val = doffdiagval[j];
            dot += val * dx[ dcolind[j] ];
        }
        dy[ row ] =  dot *alpha + beta * dy[ row ];
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    A is a matrix in mixed precision, i.e. the diagonal values are stored in
    high precision, the offdiagonal values in low precision.
    The input format is a CSR (val, row, col) in FloatComplex storing all 
    offdiagonal elements and an array containing the diagonal values in 
    DoubleComplex.
    
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
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    ddiagval    magmaDoubleComplex_ptr
                array containing diagonal values of A in DoubleComplex
                
    @param[in]
    doffdiagval magmaFloatComplex_ptr
                array containing offdiag values of A in CSR

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
magma_zcgecsrmv_mixed_prec(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr ddiagval,
    magmaFloatComplex_ptr doffdiagval,
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
    DPCT1049:197: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           zcgecsrmv_mixed_prec_kernel(
                               m, n, alpha, ddiagval, doffdiagval, drowptr,
                               dcolind, dx, beta, dy, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


