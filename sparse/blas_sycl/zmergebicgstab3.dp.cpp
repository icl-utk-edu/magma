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


// These routines merge multiple kernels from zmergebicgstab into one
// The difference to zmergedbicgstab2 is that the SpMV is not merged into the
// kernes. This results in higher flexibility at the price of lower performance.

/* -------------------------------------------------------------------------- */

void
magma_zbicgmerge1_kernel(  
    int n, 
    magmaDoubleComplex * skp,
    magmaDoubleComplex * v, 
    magmaDoubleComplex * r, 
    magmaDoubleComplex * p ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex beta=skp[1];
    magmaDoubleComplex omega=skp[2];
    if ( i<n ) {
        p[i] =  r[i] + beta * ( p[i] - omega * v[i] );
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = beta*p
    p = p-omega*beta*v
    p = p+r
    
    -> p = r + beta * ( p - omega * v ) 

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDoubleComplex_ptr
                set of scalar parameters

    @param[in]
    v           magmaDoubleComplex_ptr
                input vector v

    @param[in]
    r           magmaDoubleComplex_ptr
                input vector r

    @param[in,out]
    p           magmaDoubleComplex_ptr 
                input/output vector p

    @param[in]
    queue       magma_queue_t
                queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge1(  
    magma_int_t n, 
    magmaDoubleComplex_ptr skp,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r, 
    magmaDoubleComplex_ptr p,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, BLOCK_SIZE));
    /*
    DPCT1049:190: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgmerge1_kernel(n, skp, v, r, p, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

void
magma_zbicgmerge2_kernel(  
    int n, 
    magmaDoubleComplex * skp, 
    magmaDoubleComplex * r,
    magmaDoubleComplex * v, 
    magmaDoubleComplex * s ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex alpha=skp[0];
    if ( i < n ) {
        s[i] =  r[i] - alpha * v[i];
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    s=r
    s=s-alpha*v
        
    -> s = r - alpha * v

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDoubleComplex_ptr 
                set of scalar parameters

    @param[in]
    r           magmaDoubleComplex_ptr 
                input vector r

    @param[in]
    v           magmaDoubleComplex_ptr 
                input vector v

    @param[out]
    s           magmaDoubleComplex_ptr 
                output vector s

    @param[in]
    queue       magma_queue_t
                queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge2(  
    magma_int_t n, 
    magmaDoubleComplex_ptr skp, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr s,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, BLOCK_SIZE));

    /*
    DPCT1049:191: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgmerge2_kernel(n, skp, r, v, s, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

void
magma_zbicgmerge3_kernel(  
    int n, 
    magmaDoubleComplex * skp, 
    magmaDoubleComplex * p,
    magmaDoubleComplex * se,
    magmaDoubleComplex * t,
    magmaDoubleComplex * x, 
    magmaDoubleComplex * r
    ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex alpha=skp[0];
    magmaDoubleComplex omega=skp[2];
    if ( i<n ) {
        magmaDoubleComplex s;
        s = se[i];
        x[i] = x[i] + alpha * p[i] + omega * s;
        r[i] = s - omega * t[i];
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x=x+alpha*p
    x=x+omega*s
    r=s
    r=r-omega*t
        
    -> x = x + alpha * p + omega * s
    -> r = s - omega * t

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDoubleComplex_ptr 
                set of scalar parameters

    @param[in]
    p           magmaDoubleComplex_ptr 
                input p

    @param[in]
    s           magmaDoubleComplex_ptr 
                input s

    @param[in]
    t           magmaDoubleComplex_ptr 
                input t

    @param[in,out]
    x           magmaDoubleComplex_ptr 
                input/output x

    @param[in,out]
    r           magmaDoubleComplex_ptr 
                input/output r

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge3(  
    magma_int_t n, 
    magmaDoubleComplex_ptr skp,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, BLOCK_SIZE));
    /*
    DPCT1049:192: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zbicgmerge3_kernel(n, skp, p, s, t, x, r, item_ct1);
            });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

void
magma_zbicgmerge4_kernel_1(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if ( i==0 ) {
        magmaDoubleComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

void
magma_zbicgmerge4_kernel_2(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if ( i==0 ) {
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

void
magma_zbicgmerge4_kernel_3(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if ( i==0 ) {
        magmaDoubleComplex tmp1 = skp[4]/skp[3];
        magmaDoubleComplex tmp2 = skp[0] / skp[2];
        skp[1] =  tmp1*tmp2;
        //skp[1] =  skp[4]/skp[3] * skp[0] / skp[2];
    }
}

/**
    Purpose
    -------

    Performs some parameter operations for the BiCGSTAB with scalars on GPU.

    Arguments
    ---------

    @param[in]
    type        int
                kernel type

    @param[in,out]
    skp         magmaDoubleComplex_ptr 
                vector with parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge4(  
    magma_int_t type, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, 1);
    sycl::range<3> Gs(1, 1, 1);
    if ( type == 1 )
        /*
        DPCT1049:193: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zbicgmerge4_kernel_1(skp, item_ct1);
                           });
    else if ( type == 2 )
        /*
        DPCT1049:194: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zbicgmerge4_kernel_2(skp, item_ct1);
                           });
    else if ( type == 3 )
        /*
        DPCT1049:195: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zbicgmerge4_kernel_3(skp, item_ct1);
                           });
    else
        printf("error: no kernel called\n");

    return MAGMA_SUCCESS;
}
