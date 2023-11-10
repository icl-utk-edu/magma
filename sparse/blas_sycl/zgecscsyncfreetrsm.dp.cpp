/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/

// CSC Sync-Free SpTRSM kernel
// see paper by W. Liu, A. Li, J. D. Hogg, I. S. Duff, and B. Vinter. (2016).
// "A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves".
// 22nd International European Conference on Parallel and Distributed Computing 
// (Euro-Par '16). pp. 617-630.

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"
#include "atomicopsmagmaDoubleComplex.h"
#include <cmath>

  // for CUDA_VERSION

#define MAGMA_CSC_SYNCFREE_WARP_SIZE 32

#define MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD  0
#define MAGMA_CSC_SYNCFREE_SUBSTITUTION_BACKWARD 1

#define MAGMA_CSC_SYNCFREE_OPT_WARP_NNZ   1
#define MAGMA_CSC_SYNCFREE_OPT_WARP_RHS   2
#define MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO  3


void sptrsv_syncfree_analyser(magmaIndex_ptr         d_cscRowIdx,
                              magmaDoubleComplex_ptr d_cscVal,
                              magma_int_t            m,
                              magma_int_t            nnz,
                              magmaIndex_ptr         d_graphInDegree,
                              sycl::nd_item<3> item_ct1)
{
    const int global_id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);
    if (global_id < nnz)
    {
        dpct::atomic_fetch_add<magma_index_t,
                               sycl::access::address_space::generic_space>(
            &d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}


void sptrsm_syncfree_executor(magmaIndex_ptr         d_cscColPtr,
                              magmaIndex_ptr         d_cscRowIdx,
                              magmaDoubleComplex_ptr d_cscVal,
                              magmaIndex_ptr         d_graphInDegree,
                              magma_int_t            m,
                              magma_int_t            substitution,
                              magma_int_t            rhs,
                              magma_int_t            opt,
                              magmaDoubleComplex_ptr d_b,
                              magmaDoubleComplex_ptr d_x,
                              sycl::nd_item<3> item_ct1)
{
    const int global_id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);
    int global_x_id = global_id / MAGMA_CSC_SYNCFREE_WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                  global_x_id : m - 1 - global_x_id;

    // Initialize
    const int lane_id =
        (MAGMA_CSC_SYNCFREE_WARP_SIZE - 1) & item_ct1.get_local_id(2);

    // Prefetch
    const int pos = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    /*
    DPCT1064:73: Migrated make_cuDoubleComplex call is used in a macro
     * definition and is not valid for all macro uses. Adjust the code.
    */
    const magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    const magmaDoubleComplex coef = one / d_cscVal[pos];

    /*
    // clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (1 != d_graphInDegree[global_x_id]);
    
    // Consumer
    int graphInDegree;
    do {
        //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
        asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    }
    while (1 != graphInDegree );
    */

    for (int k = lane_id; k < rhs; k += MAGMA_CSC_SYNCFREE_WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_x[pos]) * coef;
    }

    // Producer
    const magma_index_t start_ptr =
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
              d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const magma_index_t stop_ptr  =
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
              d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;

    if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_NNZ)
    {
        for (magma_index_t jj = start_ptr + lane_id;
                           jj < stop_ptr; jj += MAGMA_CSC_SYNCFREE_WARP_SIZE)
        {
            const magma_index_t j =
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                      jj : stop_ptr - 1 - (jj - start_ptr);
            const magma_index_t rowIdx = d_cscRowIdx[j];
            for (magma_index_t k = 0; k < rhs; k++)
                atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k],
                    d_x[global_x_id * rhs + k] * d_cscVal[j]);
            /*
            DPCT1078:74: Consider replacing memory_order::acq_rel
             * with memory_order::seq_cst for correctness if strong memory order
             * restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            dpct::atomic_fetch_sub<magma_index_t,
                                   sycl::access::address_space::generic_space>(
                &d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_RHS)
    {
        for (magma_index_t jj = start_ptr; jj < stop_ptr; jj++)
        {
            const magma_index_t j =
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                      jj : stop_ptr - 1 - (jj - start_ptr);
            const magma_index_t rowIdx = d_cscRowIdx[j];
            for (magma_index_t k = lane_id;
                               k < rhs; k+=MAGMA_CSC_SYNCFREE_WARP_SIZE)
                atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k],
                    d_x[global_x_id * rhs + k] * d_cscVal[j]);
            /*
            DPCT1078:75: Consider replacing memory_order::acq_rel
             * with memory_order::seq_cst for correctness if strong memory order
             * restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            if (!lane_id) dpct::atomic_fetch_sub<
                magma_index_t, sycl::access::address_space::generic_space>(
                &d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO)
    {
        const magma_index_t len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 8) && len < 2048)
        {
            for (magma_index_t jj = start_ptr; jj < stop_ptr; jj++)
            {
                const magma_index_t j =
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                      jj : stop_ptr - 1 - (jj - start_ptr);
                const magma_index_t rowIdx = d_cscRowIdx[j];
                for (magma_index_t k = lane_id;
                                   k < rhs; k+=MAGMA_CSC_SYNCFREE_WARP_SIZE)
                    atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k],
                        d_x[global_x_id * rhs + k] * d_cscVal[j]);
                /*
                DPCT1078:76: Consider replacing
                 * memory_order::acq_rel with memory_order::seq_cst for
                 * correctness if strong memory order restrictions are needed.

                 */
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);
                if (!lane_id) dpct::atomic_fetch_sub<
                    magma_index_t, sycl::access::address_space::generic_space>(
                    &d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (magma_index_t jj = start_ptr + lane_id;
                             jj < stop_ptr; jj += MAGMA_CSC_SYNCFREE_WARP_SIZE)
            {
                const magma_index_t j = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                      jj : stop_ptr - 1 - (jj - start_ptr);
                const magma_index_t rowIdx = d_cscRowIdx[j];
                for (magma_index_t k = 0; k < rhs; k++)
                    atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k],
                        d_x[global_x_id * rhs + k] * d_cscVal[j]);
                /*
                DPCT1078:77: Consider replacing
                 * memory_order::acq_rel with memory_order::seq_cst for
                 * correctness if strong memory order restrictions are needed.

                 */
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);
                dpct::atomic_fetch_sub<
                    magma_index_t, sycl::access::address_space::generic_space>(
                    &d_graphInDegree[rowIdx], 1);
            }
        }
    }
}


extern "C" magma_int_t
magma_zgecscsyncfreetrsm_analysis(
    magma_int_t             m,
    magma_int_t             nnz,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind,
    magmaIndex_ptr          dgraphindegree,
    magmaIndex_ptr          dgraphindegree_bak,
    magma_queue_t           queue )
{
    int info = MAGMA_SUCCESS;

    int num_threads = 128;
    int num_blocks = ceil ((double)nnz / (double)num_threads);
    queue->sycl_stream()->memset(dgraphindegree, 0, m * sizeof(magma_index_t)).wait();
    queue->sycl_stream()->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           sptrsv_syncfree_analyser(drowind, dval, m, nnz,
                                                    dgraphindegree, item_ct1);
                       });

    // backup in-degree array
    queue->sycl_stream()->memcpy(dgraphindegree_bak, dgraphindegree, m * sizeof(int)).wait();
    return info;
}

extern "C" magma_int_t
magma_zgecscsyncfreetrsm_solve(
    magma_int_t             m,
    magma_int_t             nnz,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind,
    magmaIndex_ptr          dgraphindegree,
    magmaIndex_ptr          dgraphindegree_bak,
    magmaDoubleComplex_ptr  dx,
    magmaDoubleComplex_ptr  db,
    magma_int_t             substitution, 
    magma_int_t             rhs, 
    magma_queue_t           queue )
{
    int info = MAGMA_SUCCESS;

    // get an unmodified in-degree array, only for benchmarking use
    queue->sycl_stream()->memcpy(dgraphindegree, dgraphindegree_bak, m * sizeof(magma_index_t))
        .wait();

    // clear d_x for atomic operations
    queue->sycl_stream()->memset(dx, 0, sizeof(magmaDoubleComplex) * m * rhs).wait();

    int num_threads, num_blocks;

    num_threads = 4 * MAGMA_CSC_SYNCFREE_WARP_SIZE;
    num_blocks = ceil ((double)m / 
                         (double)(num_threads/MAGMA_CSC_SYNCFREE_WARP_SIZE));
    queue->sycl_stream()->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, num_threads),
                          sycl::range<3>(1, 1, num_threads)),
        [=](sycl::nd_item<3> item_ct1) {
            sptrsm_syncfree_executor(
                dcolptr, drowind, dval, dgraphindegree, m, substitution, rhs,
                MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO, db, dx, item_ct1);
        });

    return info;
}
