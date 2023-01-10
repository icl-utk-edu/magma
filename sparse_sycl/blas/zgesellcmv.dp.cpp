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


#define PRECISION_z


// SELLC SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
void 
zgesellcmv_kernel(   
    int num_rows, 
    int num_cols,
    int blocksize,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1)
{
    // threads assigned to rows
    int Idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    int offset = drowptr[item_ct1.get_group(2)];
    int border = (drowptr[item_ct1.get_group(2) + 1] - offset) / blocksize;
    if(Idx < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int n = 0; n < border; n++){
            int col =
                dcolind[offset + blocksize * n + item_ct1.get_local_id(2)];
            magmaDoubleComplex val =
                dval[offset + blocksize * n + item_ct1.get_local_id(2)];
            if( val != 0){
                  dot=dot+val*dx[col];
            }
        }

        dy[ Idx ] = dot * alpha + beta * dy [ Idx ];
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLC/SELLP.
    
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
    blocksize   magma_int_t
                number of rows in one ELL-slice

    @param[in]
    slices      magma_int_t
                number of slices in matrix

    @param[in]
    alignment   magma_int_t
                number of threads assigned to one row (=1)

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in SELLC/P

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in SELLC/P

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of SELLP

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
magma_zgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    // the kernel can only handle up to 65535 slices 
    // (~2M rows for blocksize 32)
    sycl::range<3> grid(1, 1, slices);
    magma_int_t threads = blocksize;
    /*
    DPCT1049:123: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgesellcmv_kernel(m, n, blocksize, alpha, dval,
                                             dcolind, drowptr, dx, beta, dy,
                                             item_ct1);
                       });

    return MAGMA_SUCCESS;
}
