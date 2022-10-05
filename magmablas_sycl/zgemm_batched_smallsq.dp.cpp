/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

template<int N>
void
zgemm_batched_smallsq_kernel(
        const magma_trans_t transA, magma_trans_t transB, 
        const magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, int ai, int aj, int ldda, 
                                        magmaDoubleComplex const * const * dB_array, int bi, int bj, int lddb, 
        const magmaDoubleComplex beta,  magmaDoubleComplex**               dC_array, int ci, int cj, int lddc, 
        const int batchCount, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int tz = item_ct1.get_local_id(0);
    const int bx = item_ct1.get_group(2);

    const int batchid = bx * item_ct1.get_local_range(0) + tz;
    if(batchid >= batchCount) return;
    
    const magmaDoubleComplex* __restrict__ dA = dA_array[batchid] + aj * ldda + ai;
    const magmaDoubleComplex* __restrict__ dB = dB_array[batchid] + bj * lddb + bi;
          magmaDoubleComplex* __restrict__ dC = dC_array[batchid] + cj * lddc + ci;

    /*
    DPCT1064:301: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rC = MAGMA_Z_ZERO;
    /*
    DPCT1064:302: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;

    const int slda = SLDA(N);
    const int sldb = SLDA(N);
    magmaDoubleComplex* sA = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex *sB =
        (magmaDoubleComplex *)(zdata + item_ct1.get_local_range(0) * slda * N);

    sA += tz * slda * N;
    sB += tz * sldb * N;
    
    // read A & B 
    if(transA == MagmaNoTrans){
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else{
        sA[tx * slda + ty] = (transA == MagmaTrans) ? dA[ty * ldda + tx] : MAGMA_Z_CONJ( dA[ty * ldda + tx] );
    }

    if(transB == MagmaNoTrans){
        sB[ty * sldb + tx] = dB[ty * lddb + tx];
    }
    else{
        sB[tx * sldb + ty] = (transB == MagmaTrans) ? dB[ty * lddb + tx] : MAGMA_Z_CONJ( dB[ty * lddb + tx] );
    }
    /*
    DPCT1065:300: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    /*
    DPCT1064:303: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (beta != MAGMA_Z_ZERO) {
        rC = beta * dC[ty * lddc + tx];
    }

    // multiply
    /*
    DPCT1064:304: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    rTmp = MAGMA_Z_ZERO;
#pragma unroll
    for(int j = 0; j < N; j++){
        rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
    }
    rC += alpha * rTmp;

    // write from rC
    dC[ty * lddc + tx] = rC;
}


extern "C" void 
magmablas_zgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if( !(m == n  && n == k) ){
        printf("Only square sizes are supported\n");
        info = -1;
    }

    if( m > 32){
        printf("Only square sizes of up to 32 are supported\n");
        info = -1;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
    }

    if ( m <= 0 || n <= 0 || k <= 0 ) return;
    
    magma_int_t ntcol  = magma_get_zgemm_batched_ntcol( m );
    /*
    DPCT1083:306: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    magma_int_t shmem =
        (SLDA(m) * m + SLDA(n) * n) * sizeof(magmaDoubleComplex);
                shmem *= ntcol;

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, nblocks);
    sycl::range<3> threads(ntcol, m, m);

    switch(m){
        /*
        DPCT1049:305: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<1>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:307: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<2>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:308: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<3>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:309: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<4>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:310: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<5>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:311: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<6>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:312: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<7>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:313: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<8>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:314: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 9: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<9>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:315: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 10: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<10>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:316: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 11: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<11>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:317: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 12: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<12>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:318: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 13: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<13>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:319: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 14: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<14>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:320: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 15: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<15>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:321: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<16>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:322: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 17: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<17>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:323: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 18: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<18>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:324: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 19: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<19>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:325: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 20: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<20>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:326: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 21: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<21>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:327: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 22: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<22>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:328: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 23: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<23>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:329: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 24: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<24>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:330: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 25: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<25>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:331: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 26: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<26>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:332: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 27: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<27>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:333: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 28: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<28>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:334: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 29: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<29>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:335: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 30: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<30>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:336: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 31: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<31>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:337: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 32: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgemm_batched_smallsq_kernel<32>(
                                         transA, transB, alpha, dA_array, ai,
                                         aj, ldda, dB_array, bi, bj, lddb, beta,
                                         dC_array, ci, cj, lddc, batchCount,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        default:;
    }
}
