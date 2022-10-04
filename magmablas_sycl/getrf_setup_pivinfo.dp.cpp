    /*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"

// =============================================================================
// Auxiliary routine to compute piv final destination for the current step

/******************************************************************************/
static void setup_pivinfo_devfunc(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb,
                                  sycl::nd_item<3> item_ct1)
{
    int tid = item_ct1.get_local_id(2);
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );

    // initialize pivinfo (could be done in a separate kernel using multiple thread block
    for (int s =0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS < m) && (tid < MAX_NTHREADS) )
            pivinfo[tid + s * MAX_NTHREADS] = tid + s * MAX_NTHREADS + 1;
    }
    /*
    DPCT1065:165: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tid == 0)
    {
        int i, itsreplacement, mynewrowid;
        for (i=0; i < nb; i++) {
            mynewrowid          = ipiv[i]-1; //-1 to get the index in C
            itsreplacement      = pivinfo[mynewrowid];
            pivinfo[mynewrowid] = pivinfo[i];
            pivinfo[i]          = itsreplacement;
        }
    }
}


/******************************************************************************/
static void setup_pivinfo_sm_devfunc(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb,
                                     sycl::nd_item<3> item_ct1, int *spivinfo)
{
    const int tx = item_ct1.get_local_id(2);
    const int nth = item_ct1.get_local_range(2);
        // 40 KB of shared memory

    int nchunk = magma_ceildiv( m, nth);
    int m_ = m - (nchunk-1) * nth;

    // initialize spivinfo
    for(int s = 0; s < m-nth; s+= nth){
        spivinfo[ s + tx ] = s + tx + 1;
    }
    if( tx < m_){
        spivinfo[ (nchunk-1) * nth + tx ] = (nchunk-1) * nth + tx + 1;
    }
    /*
    DPCT1065:166: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tx == 0)
    {
        int i, itsreplacement, mynewrowid;
        for (i=0; i < nb; i++) {
            mynewrowid           = ipiv[i]-1; //-1 to get the index in C
            itsreplacement       = spivinfo[mynewrowid];
            spivinfo[mynewrowid] = spivinfo[i];
            spivinfo[i]          = itsreplacement;
        }
    }
    /*
    DPCT1065:167: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // write pivinfo
    for(int s = 0; s < m-nth; s+= nth){
        pivinfo[ s + tx] = spivinfo[ s + tx ];
    }
    if( tx < m_){
        pivinfo[ (nchunk-1) * nth + tx ] = (magma_int_t)(spivinfo[ (nchunk-1) * nth + tx ]);
    }
}


/******************************************************************************/
void setup_pivinfo_kernel(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb,
                          sycl::nd_item<3> item_ct1)
{
    setup_pivinfo_devfunc(pivinfo, ipiv, m, nb, item_ct1);
}

/******************************************************************************/
void setup_pivinfo_kernel_batched(magma_int_t **pivinfo_array, magma_int_t **ipiv_array, int ipiv_offset, int m, int nb,
                                  sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(2);
    setup_pivinfo_devfunc(pivinfo_array[batchid],
                          ipiv_array[batchid] + ipiv_offset, m, nb, item_ct1);
}

/******************************************************************************/
void setup_pivinfo_kernel_vbatched(
                    magma_int_t* M, magma_int_t* N,
                    magma_int_t **pivinfo_array, int pivinfo_offset,
                    magma_int_t **ipiv_array,    int ipiv_offset,
                    int nb, sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(2);
    int my_m = (int)M[batchid];
    int my_n = (int)N[batchid];
    int my_minmn = min(my_m, my_n);
    my_m     -= ipiv_offset;
    my_minmn -= ipiv_offset;

    // check for early termination
    if(my_minmn <= 0) return;

    nb = min(nb, my_minmn);

    setup_pivinfo_devfunc(pivinfo_array[batchid] + pivinfo_offset,
                          ipiv_array[batchid] + ipiv_offset, my_m, nb,
                          item_ct1);
}


/******************************************************************************/
void setup_pivinfo_sm_kernel(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb,
                             sycl::nd_item<3> item_ct1, int *spivinfo)
{
    setup_pivinfo_sm_devfunc(pivinfo, ipiv, m, nb, item_ct1, spivinfo);
}


/******************************************************************************/
extern "C" void
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv,
                 magma_int_t m, magma_int_t nb,
                 magma_queue_t queue)
{
    if (nb == 0 ) return;
    size_t min_m_MAX_NTHREADS = min(m, MAX_NTHREADS);
    if( m > 10240 ){
        /*
        DPCT1049:168: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, min_m_MAX_NTHREADS),
                                  sycl::range<3>(1, 1, min_m_MAX_NTHREADS)),
                [=](sycl::nd_item<3> item_ct1) {
                    setup_pivinfo_kernel(pivinfo, ipiv, m, nb, item_ct1);
                });
    }
    else{
        /*
        DPCT1049:169: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    spivinfo_acc_ct1(sycl::range<1>(10240), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, min_m_MAX_NTHREADS),
                                      sycl::range<3>(1, 1, min_m_MAX_NTHREADS)),
                    [=](sycl::nd_item<3> item_ct1) {
                        setup_pivinfo_sm_kernel(pivinfo, ipiv, m, nb, item_ct1,
                                                spivinfo_acc_ct1.get_pointer());
                    });
            });
    }
}

/******************************************************************************/
extern "C" void
setup_pivinfo_batched( magma_int_t **pivinfo_array, magma_int_t **ipiv_array, magma_int_t ipiv_offset,
                         magma_int_t m, magma_int_t nb,
                         magma_int_t batchCount,
                         magma_queue_t queue)
{
    if (nb == 0 ) return;

    size_t min_m_MAX_NTHREADS = min(m, MAX_NTHREADS);
    /*
    DPCT1049:170: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount) *
                                  sycl::range<3>(1, 1, min_m_MAX_NTHREADS),
                              sycl::range<3>(1, 1, min_m_MAX_NTHREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                setup_pivinfo_kernel_batched(pivinfo_array, ipiv_array,
                                             ipiv_offset, m, nb, item_ct1);
            });
}

/******************************************************************************/
extern "C" void
setup_pivinfo_vbatched(  magma_int_t **pivinfo_array, magma_int_t pivinfo_offset,
                         magma_int_t **ipiv_array,    magma_int_t ipiv_offset,
                         magma_int_t* m, magma_int_t* n,
                         magma_int_t max_m, magma_int_t nb, magma_int_t batchCount,
                         magma_queue_t queue)
{
    if (nb == 0 ) return;

    size_t min_m_MAX_NTHREADS = min(max_m, MAX_NTHREADS);
    /*
    DPCT1049:171: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount) *
                                  sycl::range<3>(1, 1, min_m_MAX_NTHREADS),
                              sycl::range<3>(1, 1, min_m_MAX_NTHREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                setup_pivinfo_kernel_vbatched(m, n, pivinfo_array,
                                              pivinfo_offset, ipiv_array,
                                              ipiv_offset, nb, item_ct1);
            });
}

// =============================================================================
// Auxiliary routine to adjust ipiv

/******************************************************************************/
static void adjust_ipiv_devfunc(magma_int_t *ipiv, int m, int offset,
                                sycl::nd_item<3> item_ct1)
{
    int tid = item_ct1.get_local_id(2);
    if (tid < m) {
        ipiv[tid] += offset;
    }
}


/******************************************************************************/
void adjust_ipiv_kernel(magma_int_t *ipiv, int m, int offset,
                        sycl::nd_item<3> item_ct1)
{
    adjust_ipiv_devfunc(ipiv, m, offset, item_ct1);
}

/******************************************************************************/
void adjust_ipiv_kernel_batched(magma_int_t **ipiv_array, int ipiv_offset, int m, int offset,
                                sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(2);
    adjust_ipiv_devfunc(ipiv_array[batchid] + ipiv_offset, m, offset, item_ct1);
}

/******************************************************************************/
void adjust_ipiv_kernel_vbatched(
                    magma_int_t **ipiv_array, int ipiv_offset,
                    magma_int_t *minmn, int max_minmn, int offset,
                    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(2);
    int my_minmn = (int)minmn[batchid];

    if(ipiv_offset >= my_minmn) return;
    my_minmn -= ipiv_offset;
    my_minmn  = min(my_minmn, max_minmn);

    adjust_ipiv_devfunc(ipiv_array[batchid] + ipiv_offset, my_minmn, offset,
                        item_ct1);
}

/******************************************************************************/
extern "C" void
adjust_ipiv( magma_int_t *ipiv,
                 magma_int_t m, magma_int_t offset,
                 magma_queue_t queue)
{
    if (offset == 0 ) return;
    if ( m  > 1024)
    {
        fprintf( stderr, "%s: m=%lld > %lld, not supported\n",
                 __func__, (long long) m, (long long) MAX_NTHREADS );
        return;
    }
    /*
    DPCT1049:172: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, m), sycl::range<3>(1, 1, m)),
            [=](sycl::nd_item<3> item_ct1) {
                adjust_ipiv_kernel(ipiv, m, offset, item_ct1);
            });
}

/******************************************************************************/
extern "C" void
adjust_ipiv_batched( magma_int_t **ipiv_array, magma_int_t ipiv_offset,
                         magma_int_t m, magma_int_t offset,
                         magma_int_t batchCount, magma_queue_t queue)
{
    if (offset == 0 ) return;
    if ( m  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s: m=%lld > %lld, not supported\n",
                 __func__, (long long) m, (long long) MAX_NTHREADS );
        return;
    }
    /*
    DPCT1049:173: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount) *
                                             sycl::range<3>(1, 1, m),
                                         sycl::range<3>(1, 1, m)),
                       [=](sycl::nd_item<3> item_ct1) {
                           adjust_ipiv_kernel_batched(ipiv_array, ipiv_offset,
                                                      m, offset, item_ct1);
                       });
}

/******************************************************************************/
extern "C" void
adjust_ipiv_vbatched(    magma_int_t **ipiv_array, magma_int_t ipiv_offset,
                         magma_int_t *minmn, magma_int_t max_minmn, magma_int_t offset,
                         magma_int_t batchCount, magma_queue_t queue)
{
    if (offset == 0 ) return;
    if ( max_minmn  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s: m=%lld > %lld, not supported\n",
                 __func__, (long long) max_minmn, (long long) MAX_NTHREADS );
        return;
    }
    /*
    DPCT1049:174: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount) *
                                             sycl::range<3>(1, 1, max_minmn),
                                         sycl::range<3>(1, 1, max_minmn)),
                       [=](sycl::nd_item<3> item_ct1) {
                           adjust_ipiv_kernel_vbatched(ipiv_array, ipiv_offset,
                                                       minmn, max_minmn, offset,
                                                       item_ct1);
                       });
}
