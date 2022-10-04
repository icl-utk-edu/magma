/*                                                                                                     
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date                                                                        

       @precisions normal z -> s d c

       @author Stan Tomov
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"
#include "magma_templates.h"

#define NTHREADS    64
#define NBLOCKS     40

void
zdiinertia_kernel(int n, magmaDoubleComplex_const_ptr dA, int ldda, int *dneig,
                  sycl::nd_item<3> item_ct1, int *pe, int *ne, int *ze)
{
    const int tx = item_ct1.get_local_id(2);
    const int blk = item_ct1.get_group(2);
    int peig = 0, neig = 0, zeig = 0;

    // Each thread computes its part of the intertia
    for(int i=tx + blk*NTHREADS; i<n; i+= NTHREADS*NBLOCKS) {
        double diag = MAGMA_Z_REAL(dA[i+i*ldda]);
        if (diag > 0.0)
            peig++;
        else if (diag < 0.0)
            neig++;
        else
            zeig++;
    }
    
    pe[tx] = peig;
    ne[tx] = neig;
    ze[tx] = zeig;

    // The threads within a thread block sum their contributions to the inertia
    magma_sum_reduce<NTHREADS>(tx, pe, item_ct1);
    magma_sum_reduce<NTHREADS>(tx, ne, item_ct1);
    magma_sum_reduce<NTHREADS>(tx, ze, item_ct1);

    /*
    DPCT1065:251: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Attomic sum the contributions from all theread blocks (by thread 0)
    if (tx == 0){
        dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(
            &dneig[0], pe[0]);
        dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(
            &dneig[1], ne[0]);
        dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(
            &dneig[2], ze[0]);
    }   
}

/***************************************************************************//**
    Purpose
    -------
    magmablas_zdiinertia computes the inertia of a real diagonal matrix. 
    If matrix entries are complex, magmablas_zdiinertia considers the real
    part of the diagonal.                            
                        
    Arguments
    ----------
    @param[in] 
    n       INTEGER.
            On entry, N specifies the order of the matrix A. 
            N must be at least zero.
    
    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( LDDA, n ).
            The input matrix A with diagonal entries for which the inertia
            is computed. If dA is complex, the computation is done on the
            real  part of the diagonal.
 
    @param[in] 
    ldda    INTEGER.
            On entry, LDDA specifies the leading dimension of A.
            LDDA must be at least max( 1, n ). 

    @param[out]
    dneig   INTEGER array of DIMENSION 3 on the GPU memory.
            The number of positive, negative, and zero eigenvalues
            in this order.

    @param[in]
    queue   magma_queue_t. 
            Queue to execute in.

    @ingroup magma_hetrf
*******************************************************************************/ 

extern "C"
magma_int_t
magmablas_zdiinertia(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda, 
    int *dneig, 
    magma_queue_t queue )
{
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;

    if ( n < 0 ) {
        info = -1;
    } else if ( ldda < max(1, n) ) {
        info = -3;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /*
     * Quick return if possible.
     */
    if (n == 0) 
        return info;

    sycl::range<3> grid(1, 1, NBLOCKS);
    sycl::range<3> threads(1, 1, NTHREADS);

    // Set itertia to zero
    queue->cuda_stream()->memset(dneig, 0, 3 * sizeof(int));

    /*
    DPCT1049:252: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            pe_acc_ct1(sycl::range<1>(64 /*NTHREADS*/), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            ne_acc_ct1(sycl::range<1>(64 /*NTHREADS*/), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            ze_acc_ct1(sycl::range<1>(64 /*NTHREADS*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zdiinertia_kernel(n, dA, ldda, dneig, item_ct1,
                                               pe_acc_ct1.get_pointer(),
                                               ne_acc_ct1.get_pointer(),
                                               ze_acc_ct1.get_pointer());
                         });
    });

    return info;
}

// end magmablas_zdiinertia
