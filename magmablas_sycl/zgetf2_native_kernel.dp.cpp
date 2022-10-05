/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "shuffle.dp.hpp"
#include "sync.dp.hpp"
#include "atomics.dp.hpp"
#include "batched_kernel_param.h"

#define PRECISION_z

/**
    Purpose
    -------
    LU factorization of m-by-n matrix ( m >= n ).
    Each thread block caches an entire column in register.
    Thread blocks communicate and synchronize through global memory.
    Assumptions:
        1. dA is of size MxN such that N <= M.
        2. Thread block must be 1D, with TX multiple of 32 (warp size)
        3. TX must be >= n
        4. n must be less than the number of SMs on the GPU
**/

// =============================================================================
// init kernel
void
zgetf2_native_init_kernel( int n, int npages, magma_int_t *ipiv, int* update_flags,
                           sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    if( tx < n){
        ipiv[ tx ] = 0;
    }
    if( tx < max(n,npages) ){
        update_flags[ tx ] = 0;
    }
}

// =============================================================================
// the main kernel
template<int TX, int NPAGES>
void
zgetf2_native_kernel( int m, int n,
                      magmaDoubleComplex_ptr dA, int ldda,
                      volatile magma_int_t *ipiv, int gbstep,
                      volatile int* update_flag,
                      volatile magma_int_t *info, sycl::nd_item<3> item_ct1,
                      magmaDoubleComplex *sx, double *sabs, int *smax_id,
                      magmaDoubleComplex *sreg)
{
#ifdef MAGMA_HAVE_CUDA
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);
    magmaDoubleComplex rA[NPAGES] = {MAGMA_Z_ZERO};
    magmaDoubleComplex rx, rx_max;
    magmaDoubleComplex_ptr da = dA;
    int rx_id, max_id, flag = 0, linfo;
    double  rx_abs = 0.0, rx_abs_max = 0.0;
    const int m_ = m-(NPAGES-1)*TX;
    if( bx >= n ) return;

    // read the info (if it is set to non-zero a previous panel, then we don't set it again)
    linfo = (int)(*info);

    // read
    dA += bx * ldda + tx;
    #pragma unroll
    for(int i = 0; i < NPAGES-1; i++){
        rA[i] = dA[ i * TX ];
    }
    if( tx <  m_){
        rA[NPAGES-1] = dA[ (NPAGES-1) * TX ];
    }

    // main loop
    #pragma unroll
    for(int i = 0; i < n; i++){
        // izamax and write pivot for the ith thread block
        if(bx == i){
            /*
            DPCT1064:664: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rx_max = rx = (tx < i) ? MAGMA_Z_ZERO : rA[0];
            rx_abs_max = rx_abs = sycl::fabs(x()(rx)) + sycl::fabs(y()(rx));
            max_id = rx_id = tx;
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rx = rA[j];
                rx_abs = sycl::fabs(x()(rx)) + sycl::fabs(y()(rx));
                if ( rx_abs  > rx_abs_max ){
                    rx_max = rx;
                    rx_abs_max = rx_abs;
                    max_id = j * TX + tx;
                }
            }
            sx[ tx ] = rx_max;
            sabs[ tx ] = rx_abs_max;
            smax_id[ tx ] = max_id;
            /*
            DPCT1065:660: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // let the first warp do the final reduction step
            if(tx < 32){
                #pragma unroll
                for(int j = 0; j < TX; j+= 32){
                    rx     = sx[ j + tx ];
                    rx_abs = sabs[ j + tx ];
                    rx_id  = smax_id[ j + tx ];
                    if ( rx_abs  > rx_abs_max ){
                        rx_max = rx;
                        rx_abs_max = rx_abs;
                        max_id = rx_id;
                    }
                }
                magmablas_syncwarp(item_ct1);
                sx[ tx ] = rx_max;
                sabs[ tx ] = rx_abs_max;
                smax_id[ tx ] = max_id;
                magmablas_syncwarp(item_ct1);
#pragma unroll
                for(int j = 0; j < 32; j++){
                    rx     = sx[j];
                    rx_abs = sabs[j];
                    rx_id  = smax_id[j];
                    if ( rx_abs  > rx_abs_max ){
                        rx_abs_max = rx_abs;
                        rx_max = rx;
                        max_id = rx_id;
                    }
                }
            }

            if(tx == 0){
                sx[ 0 ] = rx_max;
                sabs[ 0 ] = rx_abs_max;
                smax_id[ 0 ] = (rx_abs_max == MAGMA_D_ZERO) ? i : max_id;
            }
            /*
            DPCT1065:661: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            rx_max = sx[ 0 ];
            rx_abs_max = sabs[ 0 ];
            max_id = smax_id[ 0 ];
            /*
            DPCT1065:662: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // now every thread in the i^th block has the maximum
            linfo = (rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (max_id+gbstep+1) : linfo;
            if( tx == 0){
                //printf("[%2d]: bx = %d, max_id, = %d, rx_abs_max = %f, linfo = %d\n", i, bx, max_id, rx_abs_max, linfo);
                magmablas_iatomic_exchange((magma_int_t*)info, (magma_int_t)(linfo) );
                magmablas_iatomic_exchange((magma_int_t*)&ipiv[i], (magma_int_t)(max_id+1) ); // fortran indexing
            }
            /*
            DPCT1065:663: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            //if( rx_abs_max == MAGMA_D_ZERO )return;
        }
        else{ // other thread blocks are waiting
            if(tx == 0){
                max_id = 0;
                while( max_id == 0 ){
                    max_id = ipiv[i];
                };
                smax_id[ 0 ] = max_id;
            }
            /*
            DPCT1065:665: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            max_id = smax_id[ 0 ];
            max_id -= 1; // revert fortran indexing
            linfo = (*info);
            /*
            DPCT1065:666: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            //if( (*info) != 0 ) return;
        }

        // swap
        // swap always happens between page 0 and page x
        // to avoid spilling rA to local memory, we use shared memory
        if( max_id != i){
            // all blocks swap in registers
            // for bx < i, the column is already written in memory,
            // but we have a copy in reg., so continue to swap in reg.,
            // and do one final write to memory
            #pragma unroll
            for(int j = 0; j < NPAGES; j++){
                if( j == (max_id/TX) ){
                    sx[ tx ] = rA[j];
                    /*
                    DPCT1065:667: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                    if( tx == i ){
                        magmaDoubleComplex tmp    = sx[ max_id%TX ];
                        sx[ max_id%TX ] = rA[0];
                        rA[0] = tmp;
                    }
                    /*
                    DPCT1065:668: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                    if( tx == max_id%TX ){
                        rA[j] = sx[ tx ];
                    }
                    /*
                    DPCT1065:669: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                }
            }
            //__syncthreads();
        }

        // the ith block does scal
        if(bx == i){
            /*
            DPCT1064:672: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            magmaDoubleComplex reg = (rx_max == MAGMA_Z_ZERO)
                                         ? MAGMA_Z_ONE
                                         : MAGMA_Z_DIV(MAGMA_Z_ONE, rx_max);
            // scal
            if( tx > i ){
                rA[0] *= reg;
            }
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rA[j] *= reg;
            }
            // write column i to global memory
            #pragma unroll
            for(int j = 0; j < NPAGES-1; j++){
                dA[ j * TX ] = rA[j];
            }
            if( tx <  m_){
                dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
            }
            /*
            DPCT1078:670: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            /*
            DPCT1065:671: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
                item_ct1
                    .barrier(); // after cuda 9.0, both are needed, not sure why
            if(tx == 0) magmablas_iatomic_exchange( (int *)&update_flag[ i ], 1);
        }

        // thread blocks with ID larger than i perform ger
        if(bx > i){
            if( tx == i ){
                *sreg = rA[0];
            }
            // wait for scal
            if( tx == 0){
                flag = 0;
                while( flag == 0 ){
                    flag = update_flag[ i ];
                };
            }
            /*
            DPCT1065:673: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            magmaDoubleComplex reg = *sreg;
            if( NPAGES == 1){
                if(tx > i && tx < m_){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }else{
                if(tx > i){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }
            #pragma unroll
            for(int j = 1; j < NPAGES-1; j++){
                rA[j] -= da[ i * ldda + j * TX + tx ] * reg;
            }
            if( NPAGES > 1){
                if( tx < m_ ){
                    rA[ NPAGES-1 ] -= da[ i * ldda + (NPAGES-1)*TX + tx ] * reg;
                }
            }
        }
    }

    // all blocks write their columns again except the last one
    if( bx < n-1 ){
        #pragma unroll
        for(int i = 0; i < NPAGES-1; i++){
            dA[ i * TX ] = rA[i];
        }
        if( tx <  m_){
            dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
        }
    }

#endif    // MAGMA_HAVE_CUDA
}

// =============================================================================
extern "C" magma_int_t
magma_zgetf2_native_fused(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t gbstep,
    magma_int_t *flags,
    magma_int_t *info, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const magma_int_t ntx   = ZGETF2_FUSED_NTH;

    if( m < n || m > ZGETF2_FUSED_MAX_M ){
        arginfo = -1;
    }
    else if( n > magma_getdevice_multiprocessor_count() ){
        arginfo = -2;
    }
    else if( ldda < max(1, m) ){
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t arch = magma_getdevice_arch();

    sycl::range<3> grid(1, 1, n);
    sycl::range<3> threads(1, 1, ntx);
    const magma_int_t npages = magma_ceildiv(m, ntx);
    // the kernel uses communication among thread blocks
    // as a safeguard, force one thread block per multiprocessor
    // by allocating more than half the shared memory
    magma_int_t shmem = magma_getdevice_shmem_block();
    shmem = (shmem / 2);
    int *update_flag = (int*) flags;    // update_flag is an int, not magma_int_t
    size_t max_n_npages = max(n,npages);
    /*
    DPCT1049:674: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, max_n_npages),
                                         sycl::range<3>(1, 1, max_n_npages)),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgetf2_native_init_kernel(n, npages, ipiv,
                                                     update_flag, item_ct1);
                       });
    // The case statement should cover up to ( xGETF2_CHAIN_MAX_M / ntx )
    switch(npages){
        /*
        DPCT1049:675: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 1>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:676: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 2>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:677: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 3>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:678: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 4>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:679: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 5>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:680: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 6>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:681: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 7>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:682: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 8>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:683: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 9: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 9>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:684: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 10: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 10>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:685: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 11: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 11>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:686: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 12: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 12>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:687: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 13: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 13>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:688: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 14: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 14>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:689: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 15: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 15>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:690: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 16>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:691: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 17: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 17>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:692: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 18: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 18>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:693: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 19: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 19>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:694: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 20: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sx_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sabs_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_id_acc_ct1(sycl::range<1>(ntx), cgh);
                sycl::accessor<magmaDoubleComplex, 0,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sreg_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2_native_kernel<ntx, 20>(
                                         m, n, dA, ldda, ipiv, gbstep,
                                         update_flag, info, item_ct1,
                                         sx_acc_ct1.get_pointer(),
                                         sabs_acc_ct1.get_pointer(),
                                         smax_id_acc_ct1.get_pointer(),
                                         sreg_acc_ct1.get_pointer());
                                 });
            });
            break;
#if defined(PRECISION_s) || defined(PRECISION_d)
        case 21: zgetf2_native_kernel< ntx, 21><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 22: zgetf2_native_kernel< ntx, 22><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 23: zgetf2_native_kernel< ntx, 23><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 24: zgetf2_native_kernel< ntx, 24><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 25: zgetf2_native_kernel< ntx, 25><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 26: zgetf2_native_kernel< ntx, 26><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 27: zgetf2_native_kernel< ntx, 27><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 28: zgetf2_native_kernel< ntx, 28><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 29: zgetf2_native_kernel< ntx, 29><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 30: zgetf2_native_kernel< ntx, 30><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 31: zgetf2_native_kernel< ntx, 31><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 32: zgetf2_native_kernel< ntx, 32><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 33: zgetf2_native_kernel< ntx, 33><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 34: zgetf2_native_kernel< ntx, 34><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 35: zgetf2_native_kernel< ntx, 35><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 36: zgetf2_native_kernel< ntx, 36><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 37: zgetf2_native_kernel< ntx, 37><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 38: zgetf2_native_kernel< ntx, 38><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 39: zgetf2_native_kernel< ntx, 39><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 40: zgetf2_native_kernel< ntx, 40><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 41: zgetf2_native_kernel< ntx, 41><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 42: zgetf2_native_kernel< ntx, 42><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 43: zgetf2_native_kernel< ntx, 43><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 44: zgetf2_native_kernel< ntx, 44><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 45: zgetf2_native_kernel< ntx, 45><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 46: zgetf2_native_kernel< ntx, 46><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s) || defined(PRECISION_d)
        #if defined(PRECISION_s)
        case 47: zgetf2_native_kernel< ntx, 47><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 48: zgetf2_native_kernel< ntx, 48><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 49: zgetf2_native_kernel< ntx, 49><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 50: zgetf2_native_kernel< ntx, 50><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 51: zgetf2_native_kernel< ntx, 51><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 52: zgetf2_native_kernel< ntx, 52><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 53: zgetf2_native_kernel< ntx, 53><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 54: zgetf2_native_kernel< ntx, 54><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 55: zgetf2_native_kernel< ntx, 55><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 56: zgetf2_native_kernel< ntx, 56><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 57: zgetf2_native_kernel< ntx, 57><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 58: zgetf2_native_kernel< ntx, 58><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 59: zgetf2_native_kernel< ntx, 59><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 60: zgetf2_native_kernel< ntx, 60><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 61: zgetf2_native_kernel< ntx, 61><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 62: zgetf2_native_kernel< ntx, 62><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 63: zgetf2_native_kernel< ntx, 63><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 64: zgetf2_native_kernel< ntx, 64><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 65: zgetf2_native_kernel< ntx, 65><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 66: zgetf2_native_kernel< ntx, 66><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 67: zgetf2_native_kernel< ntx, 67><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 68: zgetf2_native_kernel< ntx, 68><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 69: zgetf2_native_kernel< ntx, 69><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 70: zgetf2_native_kernel< ntx, 70><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 71: zgetf2_native_kernel< ntx, 71><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 72: zgetf2_native_kernel< ntx, 72><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 73: zgetf2_native_kernel< ntx, 73><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 74: zgetf2_native_kernel< ntx, 74><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 75: zgetf2_native_kernel< ntx, 75><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 76: zgetf2_native_kernel< ntx, 76><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 77: zgetf2_native_kernel< ntx, 77><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 78: zgetf2_native_kernel< ntx, 78><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 79: zgetf2_native_kernel< ntx, 79><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 80: zgetf2_native_kernel< ntx, 80><<<grid, threads, shmem, queue->sycl_stream() >>>( m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s)
        default: printf("size not supported \n");
    }
    return 0;
}
