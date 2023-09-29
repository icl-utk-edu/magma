/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_sampleselect.h"
#include <cstdint>

#define PRECISION_z


namespace magma_sampleselect {

static void compute_abs(const magmaDoubleComplex* __restrict__ in, double* __restrict__ out, int32_t size,
                        sycl::nd_item<3> item_ct1) 
{
    auto idx = item_ct1.get_local_id(2) +
               item_ct1.get_local_range(2) * item_ct1.get_group(2);
    if (idx >= size) {
        return;
    }

    auto v = in[idx];
    out[idx] = real(v) * real(v) + imag(v) * imag(v);
}

} // namespace magma_sampleselect

using namespace magma_sampleselect;

/**
    Purpose
    -------

    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaDoubleComplex
                array containing the values

    @param[out]
    thrs        double*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsampleselect(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{    
    magma_int_t info = 0;
    magma_int_t arch = magma_getdevice_arch();

    if( arch >= 300 ) {
        magma_int_t num_blocks = magma_ceildiv(total_size, block_size);
        magma_int_t required_size = sizeof(double) * (total_size * 2 + searchtree_size)
                                    + sizeof(int32_t) * sampleselect_alloc_size(total_size);
        auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

        double* gputmp1 = (double*)*tmp_ptr;
        double* gputmp2 = gputmp1 + total_size;
        double* gputree = gputmp2 + total_size;
        double* gpuresult = gputree + searchtree_size;
        int32_t* gpuints = (int32_t*)(gpuresult + 1);

        CHECK(realloc_result);

        /*
        DPCT1049:60: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    magma_sampleselect::compute_abs(val, gputmp1, total_size,
                                                    item_ct1);
                });
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                             sycl::range<3>(1, 1, 1)),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_sampleselect::sampleselect(
                                   gputmp1, gputmp2, gputree, gpuints,
                                   total_size, subset_size, gpuresult);
                           });
        magma_dgetvector(1, gpuresult, 1, thrs, 1, queue );
        *thrs = std::sqrt(*thrs);   

    }
    else {
        printf("error: this functionality needs CUDA architecture >= 3.5\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    return info;
}

/**
    Purpose
    -------

    This routine selects an approximate threshold separating the subset_size
    smallest magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaDoubleComplex
                array containing the values

    @param[out]
    thrs        double*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsampleselect_approx(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    auto num_blocks = magma_ceildiv(total_size, block_size);
    auto local_work = (total_size + num_threads - 1) / num_threads;
    auto required_size = sizeof(double) * (total_size + searchtree_size)
             + sizeof(int32_t) * (searchtree_width * (num_grouped_blocks + 1) + 1);
    auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

    double* gputmp = (double*)*tmp_ptr;
    double* gputree = gputmp + total_size;
    uint32_t* gpubucketidx = (uint32_t*)(gputree + searchtree_size);
    int32_t* gpurankout = (int32_t*)(gpubucketidx + 1);
    int32_t* gpucounts = gpurankout + 1;
    int32_t* gpulocalcounts = gpucounts + searchtree_width;
    uint32_t bucketidx{};

    constexpr auto size = 1 << searchtree_height; // for shared mem size in sampleselect_findbucket
    CHECK(realloc_result);

    /*
    DPCT1049:61: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_sampleselect::compute_abs(
                               val, gputmp, total_size, item_ct1);
                       });
    /*
    DPCT1049:62: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sample_buffer_acc_ct1(sycl::range<1>(sample_size), cgh);
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            leaves_acc_ct1(sycl::range<1>(searchtree_width), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, sample_size),
                                           sycl::range<3>(1, 1, sample_size)),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_sampleselect::build_searchtree(
                                 gputmp, gputree, total_size, item_ct1,
                                 sample_buffer_acc_ct1.get_pointer(),
                                 leaves_acc_ct1.get_pointer());
                         });
    });
    /*
    DPCT1049:63: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            local_tree_acc_ct1(sycl::range<1>(searchtree_size), cgh);
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            local_counts_acc_ct1(sycl::range<1>(searchtree_width), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_grouped_blocks) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                magma_sampleselect::count_buckets(
                    gputmp, gputree, gpulocalcounts, total_size, local_work,
                    item_ct1, local_tree_acc_ct1.get_pointer(),
                    local_counts_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1049:64: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto num_grouped_blocks_ct2 = num_grouped_blocks;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, searchtree_width) *
                                  sycl::range<3>(1, 1, num_grouped_blocks),
                              sycl::range<3>(1, 1, num_grouped_blocks)),
            [=](sycl::nd_item<3> item_ct1) {
                magma_sampleselect::reduce_counts(gpulocalcounts, gpucounts,
                                                  num_grouped_blocks_ct2,
                                                  item_ct1);
            });
    });
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sums_acc_ct1(sycl::range<1>(size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, searchtree_width / 2),
                              sycl::range<3>(1, 1, searchtree_width / 2)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                magma_sampleselect::sampleselect_findbucket(
                    gpucounts, subset_size, gpubucketidx, gpurankout, item_ct1,
                    sums_acc_ct1.get_pointer());
            });
    });
    magma_getvector(1, sizeof(uint32_t), gpubucketidx, 1, &bucketidx, 1, queue);
    magma_dgetvector(1, gputree + searchtree_width - 1 + bucketidx, 1, thrs, 1, queue);
    *thrs = std::sqrt(*thrs);

cleanup:
    return info;
}
