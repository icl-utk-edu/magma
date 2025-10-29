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

#define PRECISION_z
#define COMPLEX
#ifdef COMPLEX
#include <complex>
#endif

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }

void 
magma_zvalinit_kernel(  
    const magma_int_t num_el, 
    magmaDoubleComplex_ptr dval,
    sycl::nd_item<3> item_ct1) 
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    if (k < num_el) {
        dval[k] = zero;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dval        magmaDoubleComplex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zvalinit_gpu(
    magma_int_t num_el,
    magmaDoubleComplex_ptr dval,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zvalinit_kernel(num_el, dval, item_ct1);
                       });

    return MAGMA_SUCCESS;
}




void 
magma_zindexinit_kernel(  
    const magma_int_t num_el, 
    magmaIndex_ptr dind,
    sycl::nd_item<3> item_ct1) 
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (k < num_el) {
        dind[k] = 0;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dind        magmaIndex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zindexinit_gpu(
    magma_int_t num_el,
    magmaIndex_ptr dind,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zindexinit_kernel(num_el, dind, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.
    
    This is the GPU version of the operation.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in]
    B           magma_z_matrix
                Input matrix 2.

    @param[out]
    U           magma_z_matrix*
                U = A \cup B. If both matrices have a nonzero value 
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t magma_zcsr_sort_gpu(magma_z_matrix *A,
                                           magma_queue_t queue) {
    magma_int_t info = 0;
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&handle);

    oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), handle,
		     A->num_rows, A->num_cols,
                     oneapi::mkl::index_base::zero, A->drow, A->dcol, MAGMA_Z_MKL_PTR(A->dval));

    oneapi::mkl::sparse::sort_matrix(*queue->sycl_stream(), handle, {}); 
    
cleanup:
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &handle);

    return info;
}
