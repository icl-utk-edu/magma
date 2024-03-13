/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 16

#define PRECISION_z


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

void
magma_zmergeblockkrylov_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex *alpha,
    magmaDoubleComplex *p, 
    magmaDoubleComplex *x ,
    sycl::nd_item<3> item_ct1)
{
    int num_vecs = num_cols;
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int vec = item_ct1.get_group(1);

    if ( row<num_rows ) {
        magmaDoubleComplex val = x[ row + vec * num_rows ];
        
        for( int j=0; j<num_vecs; j++ ){
            magmaDoubleComplex lalpha = alpha[ j * num_vecs + vec ];
            magmaDoubleComplex xval = p[ row + j * num_rows ];
            
            val += lalpha * xval;
        }
        x[ row + vec * num_rows ] = val;
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = y / rho
    y = y / rho
    w = wt / psi
    z = z / psi
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex_ptr 
                matrix containing all SKP
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                search directions
                
    @param[in,out]
    x           magmaDoubleComplex_ptr 
                approximation vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr alpha, 
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, num_cols, BLOCK_SIZE);

    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zmergeblockkrylov_kernel(
                               num_rows, num_cols, alpha, p, x, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
