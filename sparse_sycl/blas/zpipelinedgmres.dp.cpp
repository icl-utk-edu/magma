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

#define COMPLEX

#define BLOCK_SIZE 512


template< int n >
void sum_reduce( /*int n,*/ int i, double* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:376: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }
        /*
        DPCT1065:377: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }
        /*
        DPCT1065:378: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }
        /*
        DPCT1065:379: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }
        /*
        DPCT1065:380: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }
        /*
        DPCT1065:381: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }
        /*
        DPCT1065:382: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }
        /*
        DPCT1065:383: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }
        /*
        DPCT1065:384: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }
        /*
        DPCT1065:385: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }
        /*
        DPCT1065:386: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }
        /*
        DPCT1065:387: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); }
}

void
magma_zpipelined_correction( 
    int n,  
    int k,
    magmaDoubleComplex * skp, 
    magmaDoubleComplex * r,
    magmaDoubleComplex * v ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    double zz= 0.0, tmp= 0.0;

    auto temp = (magmaDoubleComplex *)dpct_local;

    temp[i] = (i < k) ? skp[i] * skp[i] : 0.0;
    /*
    DPCT1065:388: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i < 64) {
        temp[ i ] += temp[ i + 64 ];
    }
    /*
    DPCT1065:389: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i < 32) {
        /*
        DPCT1065:390: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 32]; item_ct1.barrier();
        /*
        DPCT1065:391: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 16]; item_ct1.barrier();
        /*
        DPCT1065:392: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 8]; item_ct1.barrier();
        /*
        DPCT1065:393: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 4]; item_ct1.barrier();
        /*
        DPCT1065:394: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 2]; item_ct1.barrier();
        /*
        DPCT1065:395: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        temp[i] += temp[i + 1]; item_ct1.barrier();
    }
    if ( i == 0 ) {
        tmp = MAGMA_Z_REAL( temp[ i ] );
        zz = MAGMA_Z_REAL( skp[(k)] );
        /*
        DPCT1064:396: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        skp[k] = MAGMA_Z_MAKE(sycl::sqrt(zz - tmp), 0.0);
    }
}

void
magma_zpipelined_copyscale( 
    int n,  
    int k,
    magmaDoubleComplex * skp, 
    magmaDoubleComplex * r,
    magmaDoubleComplex * v ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex rr=skp[k];

    if ( i < n ) {
        v[i] =  r[i] * MAGMA_Z_MAKE(1.0, 0.0) / rr;
    }
}

//----------------------------------------------------------------------------//

void
magma_zpipelineddznrm2_kernel( 
    int m, 
    magmaDoubleComplex * da, 
    int ldda, 
    magmaDoubleComplex * dxnorm ,
    sycl::nd_item<3> item_ct1,
    double *sum)
{
    const int i = item_ct1.get_local_id(2);
    magmaDoubleComplex_ptr dx = da + item_ct1.get_group(2) * ldda;

    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += 512 ) {
        #ifdef REAL
            re = dx[j];
            lsum += re*re;
        #else
            re = MAGMA_Z_REAL( dx[j] );
            double im = MAGMA_Z_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    sum_reduce<512>(i, sum, item_ct1);

    if (i==0)
        /*
        DPCT1064:397: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        dxnorm[item_ct1.get_group(2)] = MAGMA_Z_MAKE(sycl::sqrt(sum[0]), 0.0);
}

//----------------------------------------------------------------------------//

void
magma_zpipelinedscale( 
    int n, 
    magmaDoubleComplex * r, 
    magmaDoubleComplex * drnorm ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if ( i<n ) {
        r[i] =  r[i] * MAGMA_Z_MAKE(1.0, 0.0) / drnorm[0];
    }
}

/**
    Purpose
    -------

    Computes the correction term of the pipelined GMRES according to P. Ghysels 
    and scales and copies the new search direction
    
    Returns the vector v = r/ ( skp[k] - (sum_i=1^k skp[i]^2) ) .

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i

    @param[in]
    k           int
                # skp entries v_i^T * r ( without r )

    @param[in]
    r           magmaDoubleComplex_ptr 
                vector of length n

    @param[in]
    v           magmaDoubleComplex_ptr 
                vector of length n
                
    @param[in]
    skp         magmaDoubleComplex_ptr 
                array of parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zcopyscale(
    magma_int_t n, 
    magma_int_t k,
    magmaDoubleComplex_ptr r, 
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs(1, 1, magma_ceildiv(k, BLOCK_SIZE));
    /*
    DPCT1083:399: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    unsigned int Ms = Bs[2] * sizeof(magmaDoubleComplex);

    sycl::range<3> Gs2(1, 1, magma_ceildiv(n, BLOCK_SIZE));

    /*
    DPCT1049:398: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zpipelined_correction(n, k, skp, r, v, item_ct1,
                                            dpct_local_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1049:400: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(Gs2 * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zpipelined_copyscale(n, k, skp, r, v, item_ct1);
            });

    return MAGMA_SUCCESS;
}


extern "C" magma_int_t
magma_dznrm2scale(
    magma_int_t m, 
    magmaDoubleComplex_ptr r, 
    magma_int_t lddr, 
    magmaDoubleComplex_ptr drnorm,
    magma_queue_t queue )
{
    sycl::range<3> blocks(1, 1, 1);
    sycl::range<3> threads(1, 1, 512);
    /*
    DPCT1049:401: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(512), cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zpipelineddznrm2_kernel(
                                 m, r, lddr, drnorm, item_ct1,
                                 sum_acc_ct1.get_pointer());
                         });
    });

    sycl::range<3> Bs(1, 1, BLOCK_SIZE);
    sycl::range<3> Gs2(1, 1, magma_ceildiv(m, BLOCK_SIZE));
    /*
    DPCT1049:402: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs, Bs),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zpipelinedscale(m, r, drnorm, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
