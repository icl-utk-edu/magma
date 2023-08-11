/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel

       @precisions normal d -> s
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_sampleselect.h"

#define PRECISION_d

namespace magma_sampleselect {

constexpr auto max_value = std::numeric_limits<double>::max();

/*
 * Sorting
 */
void sort2(double* in, int32_t i, int32_t j, bool odd) {
    auto ei = in[i];
    auto ej = in[j];
    if (odd != (ej < ei)) {
        in[i] = ej;
        in[j] = ei;
    }
}

void bitonic_sort(double* in, sycl::nd_item<3> item_ct1) {
    int32_t idx = item_ct1.get_local_id(2);
    // idx has the form | high | low | where /low/ has /round/ bits
    for (int32_t round = 0; round < bitonic_cutoff_log2; ++round) {
        // the lowest bit of /high/ decides the sort order
        bool odd = idx & (1 << round);
        for (int32_t bit = 1 << round; bit != 0; bit >>= 1) {
            // idx has the form | upper | lower | where /lower/ initially
            // has /round/ bits and gradually shrink
            int32_t lower = idx & (bit - 1);
            int32_t upper = idx ^ lower;
            // we then sort the elements | upper | 0/1 | lower |
            int32_t sort_idx = lower | (upper << 1);
            if (bit >= warp_size) {
                /*
                DPCT1065:127: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            } else {
                sycl::group_barrier(item_ct1.get_sub_group());
            }
            if (idx < 1 << (bitonic_cutoff_log2 - 1)) {
                sort2(in, sort_idx, sort_idx | bit, odd);
            }
        }
    }
}

void select_bitonic_basecase(double* __restrict__ in, double* __restrict__ out, int32_t size, int32_t rank,
                             sycl::nd_item<3> item_ct1, double *data) {

    int32_t idx = item_ct1.get_local_id(2);
    data[item_ct1.get_local_id(2)] = idx < size ? in[idx] : max_value;
    /*
    DPCT1065:128: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    bitonic_sort(data, item_ct1);
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx == 0) {
        *out = data[rank];
    }
}

int32_t searchtree_traversal(const double* searchtree, double el, uint32_t amask, uint32_t& equal_mask,
                             sycl::nd_item<3> item_ct1) {
    int32_t i = 0;
    equal_mask = amask;
    auto root_splitter = searchtree[0];
    bool next_smaller = el < root_splitter;
    for (int32_t lvl = 0; lvl < searchtree_height; ++lvl) {
        bool smaller = next_smaller;
        i = 2 * i + 2 - smaller;
        next_smaller = el < searchtree[i];
        auto local_mask =
            sycl::reduce_over_group(
                item_ct1.get_sub_group(),
                (amask &
                 (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                        smaller
                    ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>()) ^
            (smaller - 1);
        equal_mask &= local_mask;
    }
    return i - (searchtree_width - 1);
}

SYCL_EXTERNAL void build_searchtree(const double *__restrict__ in,
                                    double *__restrict__ out, int32_t size,
                                    sycl::nd_item<3> item_ct1,
                                    double *sample_buffer, double *leaves) {

    auto idx = item_ct1.get_local_id(2);

    sample_buffer[idx] = in[random_pick_idx(idx, sample_size, size)];
    /*
    DPCT1065:130: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    bitonic_sort(sample_buffer, item_ct1);
    /*
    DPCT1065:131: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx < searchtree_width) {
        leaves[idx] = sample_buffer[uniform_pick_idx(idx, searchtree_width, sample_size)];
        out[idx + searchtree_width - 1] = leaves[idx];
    }
    /*
    DPCT1065:132: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx < searchtree_width - 1) {
        out[idx] = leaves[searchtree_entry(idx)];
    }
}

template<typename BucketCallback>
void ssss_impl(const double* __restrict__ in,
                          const double* __restrict__ tree,
                          int32_t size, int32_t workcount,
                          BucketCallback bucket_cb, sycl::nd_item<3> item_ct1,
                          double *local_tree) {

    // load searchtree into shared memory
    blockwise_work_local(
        searchtree_size,
        [&](int32_t i) {
        local_tree[i] = tree[i];
        },
        item_ct1);
    /*
    DPCT1065:133: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    blockwise_work(
        workcount, size,
        [&](int32_t idx, uint32_t amask, sycl::nd_item<3> item_ct1) {
        uint32_t equal_mask{};
        auto bucket_idx = searchtree_traversal(local_tree, in[idx], amask,
                                               equal_mask, item_ct1);
        // sum of block-wide counts
        bucket_cb(idx, bucket_idx, amask, equal_mask, item_ct1);
        },
        item_ct1);
}

template<bool write>
void count_buckets_impl(const double* __restrict__ in,
                                   const double* __restrict__ tree,
                                   int32_t* __restrict__ counts,
                                   uint32_t* __restrict__ oracles,
                                   int32_t size, int32_t workcount,
                                   sycl::nd_item<3> item_ct1, double *local_tree,
                                   int32_t *local_counts) {

    blockwise_work_local(
        searchtree_width,
        [&](int32_t i) {
        local_counts[i] = 0;
        },
        item_ct1);
    /*
    DPCT1065:134: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    ssss_impl(
        in, tree, size, workcount,
        [&](int32_t idx, int32_t bucket, uint32_t amask, uint32_t mask,
            sycl::nd_item<3> item_ct1) {
        if (write) {
            static_assert(searchtree_height <= 8, "can't pack bucket idx into byte");
            store_packed_bytes(oracles, amask, bucket, idx, item_ct1);
        }
        dpct::atomic_fetch_add<int32_t,
                               sycl::access::address_space::generic_space>(
            &local_counts[bucket], 1);
        },
        item_ct1, local_tree);
    /*
    DPCT1065:135: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // store the local counts grouped by block idx
    blockwise_work_local(
        searchtree_width,
        [&](int32_t i) {
        counts[i + item_ct1.get_group(2) * searchtree_width] = local_counts[i];
        },
        item_ct1);
}

SYCL_EXTERNAL void count_buckets(const double *__restrict__ in,
                                 const double *__restrict__ tree,
                                 int32_t *__restrict__ counts, int32_t size,
                                 int32_t workcount, sycl::nd_item<3> item_ct1,
                                 double *local_tree, int32_t *local_counts) {
    count_buckets_impl<false>(in, tree, counts, nullptr, size, workcount,
                              item_ct1, local_tree, local_counts);
}

void count_buckets_write(const double* __restrict__ in,
                                    const double* __restrict__ tree,
                                    int32_t* __restrict__ counts,
                                    uint32_t* __restrict__ oracles,
                                    int32_t size, int32_t workcount,
                                    sycl::nd_item<3> item_ct1,
                                    double *local_tree, int32_t *local_counts) {
    count_buckets_impl<true>(in, tree, counts, oracles, size, workcount,
                             item_ct1, local_tree, local_counts);
}

void collect_bucket_indirect(const double* __restrict__ data,
                                        const uint32_t* __restrict__ oracles_packed,
                                        const int32_t* __restrict__ prefix_sum,
                                        double* __restrict__ out,
                                        int32_t size, uint32_t* bucket_ptr,
                                        int32_t* __restrict__ atomic, int32_t workcount,
                                        sycl::nd_item<3> item_ct1,
                                        int32_t *count) {

    auto bucket = *bucket_ptr;
    if (item_ct1.get_local_id(2) == 0) {
        *count = prefix_sum[bucket + searchtree_width * item_ct1.get_group(2)];
    }
    /*
    DPCT1065:136: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    blockwise_work(workcount, size, [&](int32_t idx, uint32_t amask,
                                        sycl::nd_item<3> item_ct1) {
            auto packed = load_packed_bytes(oracles_packed, amask, idx, item_ct1);
            int32_t ofs{};
            ofs = warp_aggr_atomic_count_predicate(count, amask,
                                                   packed == bucket, item_ct1);
            if (packed == bucket) {
                out[ofs] = data[idx];
            }
        },
        item_ct1 );
}

void launch_sampleselect(double* __restrict__ in, double* __restrict__ tmp, double* __restrict__ tree,
                                    double* __restrict__ out, int32_t* __restrict__ count_tmp, int32_t size, int32_t rank) {

    // This line seems wrong... '__CUDA_ARCH' doesn't mean anything
    #if (__CUDA_ARCH >= 350)

    if (threadIdx.x != 0) {
        return;
    }

    if (size <= bitonic_cutoff) {
        select_bitonic_basecase<<<1, bitonic_cutoff>>>(in, out, size, rank);
        return;
    }

    // launch kernels:
    // sample and build searchtree
    build_searchtree<<<1, sample_size>>>(in, tree, size);

    auto local_work = (size + num_threads - 1) / num_threads;
    auto bucket_idx = (uint32_t*)count_tmp;
    auto rank_out = ((int32_t*)bucket_idx) + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto localcounts = totalcounts + searchtree_width;
    auto oracles = (uint32_t*)(localcounts + num_grouped_blocks * searchtree_width);

    // count buckets
    count_buckets_write<<<num_grouped_blocks, block_size>>>(in, tree, localcounts, oracles, size, local_work);
    prefix_sum_counts<<<searchtree_width, num_grouped_blocks>>>(localcounts, totalcounts, num_grouped_blocks);
    sampleselect_findbucket<<<1, searchtree_width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
    collect_bucket_indirect<<<num_grouped_blocks, block_size>>>(in, oracles, localcounts, tmp, size, bucket_idx, nullptr, local_work);
    sampleselect_tailcall<<<1, 1>>>(tmp, in, tree, count_tmp, out);
#endif
}

void sampleselect_tailcall(double* __restrict__ in, double* __restrict__ tmp, double* __restrict__ tree,
                                      int32_t* __restrict__ count_tmp, double* __restrict__ out,
                                      sycl::nd_item<3> item_ct1) {
    if (item_ct1.get_local_id(2) != 0) {
        return;
    }
    auto bucket_idx = count_tmp;
    auto rank_out = bucket_idx + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;

    auto size = totalcounts[*bucket_idx];
    auto rank = *rank_out;
    launch_sampleselect(in, tmp, tree, out, count_tmp, size, rank);
}

SYCL_EXTERNAL void sampleselect(double *__restrict__ in,
                                double *__restrict__ tmp,
                                double *__restrict__ tree,
                                int32_t *__restrict__ count_tmp, int32_t size,
                                int32_t rank, double *__restrict__ out) {

    launch_sampleselect(in, tmp, tree, out, count_tmp, size, rank);
}



/**  No-Dynamic-Parallelism version **/

void launch_sampleselect_nodp(sycl::queue *stream, double *__restrict__ in,
                              double *__restrict__ tmp,
                              double *__restrict__ tree,
                              double *__restrict__ out,
                              int32_t *__restrict__ count_tmp, int32_t size,
                              int32_t rank) {

    if (size <= bitonic_cutoff) {
        /*
        DPCT1049:141: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler &cgh) {
            sycl::accessor<double, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                data_acc_ct1(sycl::range<1>(bitonic_cutoff), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, bitonic_cutoff),
                                  sycl::range<3>(1, 1, bitonic_cutoff)),
                [=](sycl::nd_item<3> item_ct1) {
                    select_bitonic_basecase(in, out, size, rank, item_ct1,
                                            data_acc_ct1.get_pointer());
                });
        });
        stream->wait();
        return;
    }

    // launch kernels:
    // sample and build searchtree
    /*
    DPCT1049:137: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sample_buffer_acc_ct1(sycl::range<1>(sample_size), cgh);
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            leaves_acc_ct1(sycl::range<1>(searchtree_width), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, sample_size),
                                           sycl::range<3>(1, 1, sample_size)),
                         [=](sycl::nd_item<3> item_ct1) {
                             build_searchtree(
                                 in, tree, size, item_ct1,
                                 sample_buffer_acc_ct1.get_pointer(),
                                 leaves_acc_ct1.get_pointer());
                         });
    });

    auto local_work = (size + num_threads - 1) / num_threads;
    auto bucket_idx = (uint32_t*)count_tmp;
    auto rank_out = ((int32_t*)bucket_idx) + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto localcounts = totalcounts + searchtree_width;
    auto oracles = (uint32_t*)(localcounts + num_grouped_blocks * searchtree_width);

    // count buckets
    /*
    DPCT1049:138: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
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
                count_buckets_write(in, tree, localcounts, oracles, size,
                                    local_work, item_ct1,
                                    local_tree_acc_ct1.get_pointer(),
                                    local_counts_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1049:139: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        auto num_grouped_blocks_ct2 = num_grouped_blocks;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, searchtree_width) *
                                  sycl::range<3>(1, 1, num_grouped_blocks),
                              sycl::range<3>(1, 1, num_grouped_blocks)),
            [=](sycl::nd_item<3> item_ct1) {
                prefix_sum_counts(localcounts, totalcounts,
                                  num_grouped_blocks_ct2, item_ct1);
            });
    });
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sums_acc_ct1(sycl::range<1>(size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, searchtree_width / 2),
                              sycl::range<3>(1, 1, searchtree_width / 2)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                sampleselect_findbucket(totalcounts, rank, bucket_idx, rank_out,
                                        item_ct1, sums_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1049:140: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<int32_t, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            count_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_grouped_blocks) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                collect_bucket_indirect(in, oracles, localcounts, tmp, size,
                                        bucket_idx, nullptr, local_work,
                                        item_ct1, count_acc_ct1.get_pointer());
            });
    });

    sampleselect_tailcall_nodp(stream, tmp, in, tree, count_tmp, out);
}

void sampleselect_tailcall_nodp(sycl::queue *stream, double *__restrict__ in,
                                double *__restrict__ tmp,
                                double *__restrict__ tree,
                                int32_t *__restrict__ count_tmp,
                                double *__restrict__ out) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    //if (threadIdx.x != 0) {
    //    return;
    //}
    int32_t* bucket_idx = count_tmp;
    int32_t* rank_out = bucket_idx + 1;
    int32_t* atomic = rank_out + 1;
    int32_t* totalcounts = atomic + 1;
    /* Load the size/rank */
    /*`
    int32_t size = totalcounts[*bucket_idx];
    int32_t rank = *rank_out;
    */
    int32_t bi0, size, rank;

    /* Load indexes */

    stream->wait();
    q_ct1.memcpy((void *)&bi0, bucket_idx, sizeof(bi0));
    q_ct1.memcpy((void *)&size, totalcounts + bi0, sizeof(size));
    q_ct1.memcpy((void *)&rank, rank_out, sizeof(rank)).wait();

    launch_sampleselect_nodp(stream, in, tmp, tree, out, count_tmp, size, rank);
}

void sampleselect_nodp(sycl::queue *stream, double *__restrict__ in,
                       double *__restrict__ tmp, double *__restrict__ tree,
                       int32_t *__restrict__ count_tmp, int32_t size,
                       int32_t rank, double *__restrict__ out) {

        launch_sampleselect_nodp(stream, in, tmp, tree, out, count_tmp, size, rank);
}




} // namespace magma_sampleselect
