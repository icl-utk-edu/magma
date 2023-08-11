/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from bicgstab into one.

/* -------------------------------------------------------------------------- */

void
magma_zbicgstab_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex omega,
    magmaDoubleComplex *r, 
    magmaDoubleComplex *v,
    magmaDoubleComplex *p ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = r[ i+j*num_rows ] + 
                beta * ( p[ i+j*num_rows ] - omega * v[ i+j*num_rows ] );
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = r + beta * ( p - omega * v )
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        magmaDoubleComplex
                scalar
                
    @param[in]
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    r           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    p           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr r, 
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr p,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:48: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgstab_1_kernel(num_rows, num_cols, beta,
                                                    omega, r, v, p, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zbicgstab_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr s ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            s[ i+j*num_rows ] = r[ i+j*num_rows ] - alpha * v[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    s = r - alpha v

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar
                
    @param[in]
    r           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    s           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr s, 
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:49: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgstab_2_kernel(num_rows, num_cols, alpha,
                                                    r, v, s, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zbicgstab_3_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex *p,
    magmaDoubleComplex *s,
    magmaDoubleComplex *t,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp = s[ i+j*num_rows ];
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * p[ i+j*num_rows ] + omega * tmp;
            r[ i+j*num_rows ] = tmp - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * p + omega * s
    r = s - omega * t

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar
                
    @param[in]
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    s           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    t           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    x           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:50: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zbicgstab_3_kernel(num_rows, num_cols, alpha, omega, p, s,
                                         t, x, r, item_ct1);
            });

    return MAGMA_SUCCESS;
}


void
magma_zbicgstab_4_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex *y,
    magmaDoubleComplex *z,
    magmaDoubleComplex *s,
    magmaDoubleComplex *t,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * y[ i+j*num_rows ] + omega * z[ i+j*num_rows ];
            r[ i+j*num_rows ] = s[ i+j*num_rows ] - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * y + omega * z
    r = s - omega * t

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar
                
    @param[in]
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    z           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    s           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    t           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    x           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:51: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zbicgstab_4_kernel(num_rows, num_cols, alpha, omega, y, z,
                                         s, t, x, r, item_ct1);
            });

    return MAGMA_SUCCESS;
}
