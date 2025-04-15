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


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

void
magma_zqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex *y, 
    magmaDoubleComplex *z,
    magmaDoubleComplex *v,
    magmaDoubleComplex *w ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
            magmaDoubleComplex ztmp = z[ i+j*num_rows ] / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
        }
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
    rho         magmaDoubleComplex
                scalar
                
    @param[in]
    psi         magmaDoubleComplex
                scalar
                
    @param[in,out]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    z           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    w           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:142: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_1_kernel(num_rows, num_cols, rho, psi, y,
                                               z, v, w, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex pde,
    magmaDoubleComplex rde,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr p, 
    magmaDoubleComplex_ptr q ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = y[ i+j*num_rows ] - pde * p[ i+j*num_rows ];
            q[ i+j*num_rows ] = z[ i+j*num_rows ] - rde * q[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = y - pde * p
    q = z - rde * q

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    pde         magmaDoubleComplex
                scalar

    @param[in]
    rde         magmaDoubleComplex
                scalar
                
    @param[in]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    z           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    p           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    q           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex pde,
    magmaDoubleComplex rde,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr p, 
    magmaDoubleComplex_ptr q, 
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:143: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_2_kernel(num_rows, num_cols, pde, rde, y,
                                               z, p, q, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_3_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex beta,
    magmaDoubleComplex *pt,
    magmaDoubleComplex *v,
    magmaDoubleComplex *y ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp = pt[ i+j*num_rows ] - beta * v[ i+j*num_rows ];
            v[ i+j*num_rows ] = tmp;
            y[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = pt - beta * v
    y = v

    Arguments
    ---------

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
    pt          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    y           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr y,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:144: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_3_kernel(num_rows, num_cols, beta, pt, v,
                                               y, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_4_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex eta,
    magmaDoubleComplex *p,
    magmaDoubleComplex *pt,
    magmaDoubleComplex *d,
    magmaDoubleComplex *s,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmpd = eta * p[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            magmaDoubleComplex tmps = eta * pt[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p;
    s = eta * pt;
    x = x + d;
    r = r - s;

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaDoubleComplex
                scalar
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    pt          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    d           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    s           magmaDoubleComplex_ptr 
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
magma_zqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr d, 
    magmaDoubleComplex_ptr s, 
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:145: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_4_kernel(num_rows, num_cols, eta, p, pt,
                                               d, s, x, r, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_5_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex eta,
    magmaDoubleComplex pds,
    magmaDoubleComplex *p,
    magmaDoubleComplex *pt,
    magmaDoubleComplex *d,
    magmaDoubleComplex *s,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmpd = eta * p[ i+j*num_rows ] + pds * d[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            magmaDoubleComplex tmps = eta * pt[ i+j*num_rows ] + pds * s[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p + pds * d;
    s = eta * pt + pds * s;
    x = x + d;
    r = r - s;

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaDoubleComplex
                scalar
                    
    @param[in]
    pds         magmaDoubleComplex
                scalar
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    pt          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    d           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    s           magmaDoubleComplex_ptr 
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
magma_zqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex pds,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr d, 
    magmaDoubleComplex_ptr s, 
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:146: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_5_kernel(num_rows, num_cols, eta, pds, p,
                                               pt, d, s, x, r, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_6_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex *y, 
    magmaDoubleComplex *z,
    magmaDoubleComplex *v,
    magmaDoubleComplex *w,
    magmaDoubleComplex *wt ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex wttmp = wt[ i+j*num_rows ]
                                - MAGMA_Z_CONJ( beta ) * w[ i+j*num_rows ];
                                
            wt[ i+j*num_rows ] = wttmp;
            
            magmaDoubleComplex ztmp = wttmp / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
            magmaDoubleComplex ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:
    
    wt = wt - conj(beta) * w
    v = y / rho
    y = y / rho
    w = wt / psi
    z = wt / psi
    
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
    rho         magmaDoubleComplex
                scalar
                
    @param[in]
    psi         magmaDoubleComplex
                scalar
                
    @param[in,out]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    z           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    w           magmaDoubleComplex_ptr 
                vector
                    
    @param[in,out]
    wt          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magmaDoubleComplex_ptr wt,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:147: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_6_kernel(num_rows, num_cols, beta, rho,
                                               psi, y, z, v, w, wt, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void
magma_zqmr_7_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex beta,
    magmaDoubleComplex *pt,
    magmaDoubleComplex *v,
    magmaDoubleComplex *vt ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp = pt[ i+j*num_rows ] - beta * v[ i+j*num_rows ];
            vt[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    vt = pt - beta * v

    Arguments
    ---------

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
    pt          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    vt          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_7(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr vt,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:148: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_7_kernel(num_rows, num_cols, beta, pt, v,
                                               vt, item_ct1);
                       });

    return MAGMA_SUCCESS;
}



void
magma_zqmr_8_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex *vt, 
    magmaDoubleComplex *wt,
    magmaDoubleComplex *y, 
    magmaDoubleComplex *z,
    magmaDoubleComplex *v,
    magmaDoubleComplex *w ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            y[ i+j*num_rows ] = y[ i+j*num_rows ] / rho;
            v[ i+j*num_rows ] = vt[ i+j*num_rows ] / rho;
            z[ i+j*num_rows ] = z[ i+j*num_rows ] / psi;
            w[ i+j*num_rows ] = wt[ i+j*num_rows ] / psi;
        }
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
    rho         magmaDoubleComplex
                scalar
                
    @param[in]
    psi         magmaDoubleComplex
                scalar
                
    @param[in]
    vt          magmaDoubleComplex_ptr 
                vector

    @param[in]
    wt          magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    z           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    w           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zqmr_8(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr vt,
    magmaDoubleComplex_ptr wt,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    /*
    DPCT1049:149: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zqmr_8_kernel(num_rows, num_cols, rho, psi, vt,
                                               wt, y, z, v, w, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
