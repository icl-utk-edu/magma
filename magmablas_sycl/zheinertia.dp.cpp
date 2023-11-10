/*                                                                                                     
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date                                                                        

       @author Hadeer Farahat
       @author Stan Tomov

       @precisions normal z -> s d c 
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_d.h"
#include "magma_templates.h"

#define NTHREADS    128
#define NBLOCKS      40

void
zheinertia_upper_kernel(int n, magmaDoubleComplex_const_ptr dA, int ldda, magma_int_t *ipiv, int *dneig,
                        sycl::nd_item<3> item_ct1, int *pe, int *ne, int *ze)
{
    const int tx = item_ct1.get_local_id(2);
    const int blk = item_ct1.get_group(2);
    int peig = 0, neig = 0, zeig = 0;
    double diag, t=0.0;
    int i = 0, k, nk, count, sc = sycl::ceil((double)n / (NBLOCKS * NTHREADS));

    // unrolling iteration i=0
    k = (tx + blk*NTHREADS)*sc;
    if (k<n && k-1>=0 )
        if ( ipiv[k]-1 < 0 && ipiv[k-1] == ipiv[k]) {
            count =1;
            nk = k-2;
            // check all the previous pivot values 
            while (nk >=0 && ipiv[nk] == ipiv[k] ){
                count ++;
                nk--;
            }
            // if count is odd, it means that the current pivot is a second element of a 2-by-2 diagonal block
            if ( count%2 == 1 ){
                diag = MAGMA_Z_ABS(dA[(k-1)+k*ldda]);
                t=0.0;
                if (diag > 0.) 
                    peig++;
                else if (diag < 0.)
                    neig++;
                else
                    zeig++;
                i = 1;
            }
        }
    
    // Each thread computes its part of the intertia (sc columns)
    #pragma unroll
    for(i=i; i<sc; i++){
        k=((tx + blk*NTHREADS)*sc)+i;
        if (k>=n)
            break;
        diag = MAGMA_Z_REAL(dA[k+k*ldda]);
        if (ipiv[k]-1 < 0){   
            if (t != 0.) {
                diag = t;
                t = 0.;
            } else {
                t = MAGMA_Z_ABS( dA[k+(k+1)*ldda] ); 
                diag = (diag/t) * MAGMA_Z_REAL( dA[(k+1)*(1+ldda)] ) - t; 
            }
        }

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
    DPCT1065:805: Consider replacing sycl::nd_item::barrier() with
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

void
zheinertia_lower_kernel(int n, magmaDoubleComplex_const_ptr dA, int ldda, magma_int_t *ipiv, int *dneig,
                        sycl::nd_item<3> item_ct1, int *pe, int *ne, int *ze)
{
    const int tx = item_ct1.get_local_id(2);
    const int blk = item_ct1.get_group(2);
    int peig = 0, neig = 0, zeig = 0;
    double diag, t=0.0;
    int i = 0, k, nk, count, sc = sycl::ceil((double)n / (NBLOCKS * NTHREADS));

    // unrolling iteration i=0
    k = (tx + blk*NTHREADS)*sc;
    if (k<n && k-1>=0 )
        if ( ipiv[k]-1 < 0 && ipiv[k-1] == ipiv[k]) {
            count =1;
            nk = k-2;
            // check all the previous pivot values
            while (nk >=0 && ipiv[nk] == ipiv[k] ){
                count ++;
                nk--;
            }
            // if count is odd, it means that the current pivot is a second element of a 2-by-2 diagonal block
            if ( count%2 == 1 ){
                diag = MAGMA_Z_ABS(dA[(k-1)*ldda+k]);
                t=0.0;
                if (diag > 0.)
                    peig++;
                else if (diag < 0.)
                    neig++;
                else
                    zeig++;
                i = 1;
            }
        }

    // Each thread computes its part of the intertia (sc columns)
    #pragma unroll
    for(i=i; i<sc; i++){
        k=((tx + blk*NTHREADS)*sc)+i;
        if (k>=n)
            break;
        diag = MAGMA_Z_REAL(dA[k+k*ldda]);
        if (ipiv[k]-1 < 0){
            if (t != 0.) {
                diag = t;
                t = 0.;
            } else {
                t = MAGMA_Z_ABS( dA[k*ldda+(k+1)] );
                diag = (diag/t) * MAGMA_Z_REAL( dA[(k+1)*(1+ldda)] ) - t;
            }
        }

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
    DPCT1065:806: Consider replacing sycl::nd_item::barrier() with
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
    magmablas_zheinertia computes the inertia of a hermitian and block 
    diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks. These are matrices
    comming from the Bunch-Kaufman with diagonal pivoting factorizations 
    (the ZHETRF routine). 
                        
    Arguments
    ----------
    @param[in] 
    n       INTEGER.
            On entry, N specifies the order of the matrix A. 
            N must be at least zero.
    
    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( LDDA, n ).
            The input matrix A with 1-by-1 and 2-by-2 diagonal block entries 
            for which the inertia is computed. 
 
    @param[in] 
    ldda    INTEGER.
            On entry, LDDA specifies the leading dimension of A.
            LDDA must be at least max( 1, n ). 

    @param[in]
    ipiv    INTEGER array, dimension (N) 
            The pivot vector from dsytrf.

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
magmablas_zheinertia(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda, 
    magma_int_t *ipiv,
    int *dneig, 
    magma_queue_t queue )
{
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;
    bool upper = (uplo == MagmaUpper);
    if (! upper && uplo != MagmaLower) {
        info = -1;
    } else if ( n < 0 ) {
        info = -2;
    } else if ( ldda < max(1, n) ) {
        info = -4;
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
    queue->sycl_stream()->memset(dneig, 0, 3 * sizeof(int));

    if (upper)
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    pe_acc_ct1(sycl::range<1>(NTHREADS), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ne_acc_ct1(sycl::range<1>(NTHREADS), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ze_acc_ct1(sycl::range<1>(NTHREADS), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zheinertia_upper_kernel(
                                         n, dA, ldda, ipiv, dneig, item_ct1,
                                         pe_acc_ct1.get_pointer(),
                                         ne_acc_ct1.get_pointer(),
                                         ze_acc_ct1.get_pointer());
                                 });
            });
    else
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    pe_acc_ct1(sycl::range<1>(NTHREADS), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ne_acc_ct1(sycl::range<1>(NTHREADS), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ze_acc_ct1(sycl::range<1>(NTHREADS), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zheinertia_lower_kernel(
                                         n, dA, ldda, ipiv, dneig, item_ct1,
                                         pe_acc_ct1.get_pointer(),
                                         ne_acc_ct1.get_pointer(),
                                         ze_acc_ct1.get_pointer());
                                 });
            });

    return info;
}

// end magmablas_zheinertia
