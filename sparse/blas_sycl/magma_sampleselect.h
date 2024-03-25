/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>
#include "magmasparse_internal.h"
#undef max

#define DEFAULT_WIDTH 32
namespace magma_sampleselect {

constexpr int32_t sample_size_log2 = 10;
constexpr int32_t sample_size = 1 << sample_size_log2;
constexpr int32_t searchtree_height = 8;
constexpr int32_t searchtree_width = 1 << searchtree_height;
constexpr int32_t searchtree_size = 2 * searchtree_width - 1;

constexpr int32_t warp_size_log2 = 5;
constexpr int32_t warp_size = 1 << warp_size_log2;
constexpr int32_t max_block_size_log2 = 9;
constexpr int32_t max_block_size = 1 << max_block_size_log2;

constexpr int32_t bitonic_cutoff_log2 = sample_size_log2;
constexpr int32_t bitonic_cutoff = 1 << bitonic_cutoff_log2;

constexpr auto block_size = max_block_size;
constexpr auto num_grouped_blocks = block_size;
constexpr auto num_threads = block_size * num_grouped_blocks;

constexpr uint32_t full_mask = 0xffffffff;

/*
 * Forward declarations
 */
int32_t sampleselect_alloc_size(int32_t);
int32_t realloc_if_necessary(magma_ptr*,magma_int_t*,magma_int_t);

/*
 * Type-independent kernels
 */
SYCL_EXTERNAL void reduce_counts(const int32_t *, int32_t *, int32_t,
                                 sycl::nd_item<3> item_ct1);
SYCL_EXTERNAL void prefix_sum_counts(int32_t *, int32_t *, int32_t,
                                     sycl::nd_item<3> item_ct1);
SYCL_EXTERNAL void sampleselect_findbucket(int32_t *, int32_t, uint32_t *,
                                           int32_t *, sycl::nd_item<3> item_ct1,
                                           int32_t *sums);

/*
 * Type-dependent kernels
 */

void build_searchtree(const float*,float*,int32_t);
SYCL_EXTERNAL void build_searchtree(const double *, double *, int32_t,
                                    sycl::nd_item<3> item_ct1,
                                    double *sample_buffer, double *leaves);
SYCL_EXTERNAL void build_searchtree(const float *, float *, int32_t,
                                    sycl::nd_item<3> item_ct1,
                                    float *sample_buffer, float *leaves);
void select_bitonic_basecase(float*,float*,int32_t,int32_t);
void select_bitonic_basecase(double*,double*,int32_t,int32_t,
                             sycl::nd_item<3> item_ct1, double *data);
void sampleselect(float*,float*,float*,int32_t*,int32_t,int32_t,float*);
SYCL_EXTERNAL void sampleselect(double *, double *, double *, int32_t *,
                                int32_t, int32_t, double *);
SYCL_EXTERNAL void sampleselect(float *, float *, float *, int32_t *,
                                int32_t, int32_t, float *);
void count_buckets(const float*,const float*,int32_t*,int32_t,int32_t);
SYCL_EXTERNAL void count_buckets(const double *, const double *, int32_t *,
                                 int32_t, int32_t, sycl::nd_item<3> item_ct1,
                                 double *local_tree, int32_t *local_counts);
SYCL_EXTERNAL void count_buckets(const float *, const float *, int32_t *,
                                 int32_t, int32_t, sycl::nd_item<3> item_ct1,
                                 float *local_tree, int32_t *local_counts);
void count_buckets_write(const float*,const float*,int32_t*,uint32_t*,int32_t,int32_t);
void count_buckets_write(const double*,const double*,int32_t*,uint32_t*,int32_t,int32_t,
                         sycl::nd_item<3> item_ct1, double *local_tree,
                         int32_t *local_counts);
void collect_bucket_indirect(const float*,const uint32_t*,const int32_t*,float*,int32_t,uint32_t*,int32_t*,int32_t);
void collect_bucket_indirect(const double*,const uint32_t*,const int32_t*,double*,int32_t,uint32_t*,int32_t*,int32_t,
                             sycl::nd_item<3> item_ct1, int32_t *count);

void sampleselect_tailcall(float*,float*,float*,int32_t*,float*);
void sampleselect_tailcall(double*,double*,double*,int32_t*,double*,
                           sycl::nd_item<3> item_ct1);


/* No-DP versions */
void sampleselect_nodp(sycl::queue *stream, float *, float *, float *,
                       int32_t *, int32_t, int32_t, float *);
void sampleselect_nodp(sycl::queue *stream, double *, double *, double *,
                       int32_t *, int32_t, int32_t, double *);
void sampleselect_tailcall_nodp(sycl::queue *stream, float *, float *, float *,
                                int32_t *, float *);
void sampleselect_tailcall_nodp(sycl::queue *stream, double *, double *,
                                double *, int32_t *, double *);

/*
 * Type-independent helpers
 */

/*
 * Sampling
 */
inline int32_t uniform_pick_idx(int32_t idx, int32_t samplesize, int32_t size) {
    auto stride = size / samplesize;
    if (stride == 0) {
        return idx * size / samplesize;
    } else {
        return idx * stride + stride / 2;
    }
}

inline int32_t random_pick_idx(int32_t idx, int32_t samplesize, int32_t size) {
    // TODO
    return uniform_pick_idx(idx, samplesize, size);
}

inline int32_t prefix_popc(uint32_t mask, int32_t shift) {
    return sycl::popcount(mask << (32 - shift));
    /* alternative:
    int32_t prefix_mask = (1 << shift) - 1;
    return __popc(mask & prefix_mask);
     */
}

inline int32_t warp_aggr_atomic_count_mask(int32_t* atomic, uint32_t amask, uint32_t mask,
                                           sycl::nd_item<3> item_ct1) {
    auto lane_idx = item_ct1.get_local_id(2) % warp_size;
    int32_t ofs{};
    if (lane_idx == 0) {
      ofs = dpct::atomic_fetch_add<int32_t,
                                   sycl::access::address_space::generic_space>(
          atomic, sycl::popcount(mask));
    }
    ofs = dpct::experimental::select_from_sub_group(amask, item_ct1.get_sub_group(), ofs, 0,
		                                    DEFAULT_WIDTH);
    auto local_ofs = prefix_popc(mask, lane_idx);
    return ofs + local_ofs;
}

inline int32_t warp_aggr_atomic_count_predicate(int32_t* atomic, uint32_t amask, bool predicate,
                                                sycl::nd_item<3> item_ct1) {
    auto mask = sycl::reduce_over_group(
        item_ct1.get_sub_group(),
        (amask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                predicate
            ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    return warp_aggr_atomic_count_mask(atomic, amask, mask, item_ct1);
}

/*
 * Unaligned byte storage
 */
inline void store_packed_bytes(uint32_t* output, uint32_t amask, uint32_t byte, int32_t idx,
                               sycl::nd_item<3> item_ct1) {
    // pack 4 consecutive bytes into an integer
    uint32_t result = byte;
    // ------00 -> ----1100
    result |=
        dpct::experimental::permute_sub_group_by_xor(amask, item_ct1.get_sub_group(), result, 1, 4)
        << 8;
    // ----1100 -> 33221100
    result |=
        dpct::experimental::permute_sub_group_by_xor(amask, item_ct1.get_sub_group(), result, 2, 4)
        << 16;
    if (idx % 4 == 0) {
        output[idx / 4] = result;
    }
}

inline uint32_t load_packed_bytes(const uint32_t* input, uint32_t amask, int32_t idx,
                                  sycl::nd_item<3> item_ct1) {
    auto char_idx = idx % 4;
    auto pack_idx = idx / 4;
    uint32_t packed{};
    // first thread in quartet loads the data
    if (char_idx == 0) {
        packed = input[pack_idx];
    }
    // distribute the data onto all threads
    packed = dpct::experimental::select_from_sub_group(amask, item_ct1.get_sub_group(), packed,
                                                       (pack_idx * 4) % warp_size, 4);
    packed >>= char_idx * 8;
    packed &= 255;
    return packed;
}

/*
 * Prefix sum
 */
inline void small_prefix_sum_upward(int32_t* data, sycl::nd_item<3> item_ct1) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = item_ct1.get_local_id(2);
    // upward phase: reduce
    // here we build an implicit reduction tree, overwriting values
    // the entry at the end of a power-of-two block stores the sum of this block
    // the block sizes are increased stepwise
    for (int32_t blocksize = 2; blocksize <= size; blocksize *= 2) {
        int32_t base_idx = idx * blocksize;
        if (blocksize > warp_size || true) { //TODO rethink
            /*
            DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        } else {
            sycl::group_barrier(item_ct1.get_sub_group());
        }
        if (base_idx < size) {
            data[base_idx + blocksize - 1] += data[base_idx + blocksize / 2 - 1];
        }
    }
}

inline void small_prefix_sum_downward(int32_t* data, sycl::nd_item<3> item_ct1) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = item_ct1.get_local_id(2);
    // downward phase: build prefix sum
    // every right child stores the sum of its left sibling
    // every left child stores its own sum
    // thus we store zero at the root
    if (idx == 0) {
        data[size - 1] = 0;
    }
    for (int32_t blocksize = size; blocksize != 1; blocksize /= 2) {
        int32_t base_idx = idx * blocksize;
        if (blocksize > warp_size || true) { //TODO rethink
            static_assert(size / warp_size <= warp_size, "insufficient synchronization");
            /*
            DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        } else {
            sycl::group_barrier(item_ct1.get_sub_group());
        }
        if (base_idx < size) {
            // we preserve the invariant for the next level
            auto r = data[base_idx + blocksize - 1];
            auto l = data[base_idx + blocksize / 2 - 1];
            data[base_idx + blocksize / 2 - 1] = r;
            data[base_idx + blocksize - 1] = l + r;
        }
    }
}

inline void small_prefix_sum(int32_t* data, sycl::nd_item<3> item_ct1) {
    small_prefix_sum_upward(data, item_ct1);
    /*
    DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    small_prefix_sum_downward(data, item_ct1);
}

/*
 * Prefix sum selection
 */
inline void prefix_sum_select(const int32_t* counts, int32_t rank, uint32_t* out_bucket, int32_t* out_rank,
                              sycl::nd_item<3> item_ct1, int32_t *sums) {
    constexpr auto size = 1 << searchtree_height;
    // first compute prefix sum of counts
    auto idx = item_ct1.get_local_id(2);

    sums[2 * idx] = counts[2 * idx];
    sums[2 * idx + 1] = counts[2 * idx + 1];
    small_prefix_sum(sums, item_ct1);
    /*
    DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (idx >= warp_size) {
        return;
    }
    // then determine which group of size step the element belongs to
    constexpr int32_t step = size / warp_size;
    static_assert(step <= warp_size, "need a third selection level");
    auto mask = sycl::reduce_over_group(
        item_ct1.get_sub_group(),
        (full_mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                sums[(warp_size - idx - 1) * step] > rank
            ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    if (idx >= step) {
        return;
    }
    auto group = sycl::clz((int)mask) - 1;
    // finally determine which bucket within the group the element belongs to
    auto base_idx = step * group;
    constexpr auto cur_mask = ((1u << (step - 1)) << 1) - 1;
    mask = sycl::reduce_over_group(
        item_ct1.get_sub_group(),
        (cur_mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                sums[base_idx + (step - idx - 1)] > rank
            ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    // here we need to subtract warp_size - step since we only use a subset of the warp
    if (idx == 0) {
        *out_bucket = sycl::clz((int)mask) - 1 - (warp_size - step) + base_idx;
        *out_rank = rank - sums[*out_bucket];
    }
}

/*
 * Work assignment
 */
template<typename F, typename... Args>
inline void blockwise_work(int32_t local_work, int32_t size, F function,
                           sycl::nd_item<3> item_ct1, Args... args) {
    auto stride = num_grouped_blocks *
                  item_ct1.get_local_range(
                      2); // this should be a parameter for general cases
    auto base_idx = item_ct1.get_local_id(2) +
                    item_ct1.get_local_range(2) * item_ct1.get_group(2);
    for (int32_t i = 0; i < local_work; ++i) {
        int32_t idx = base_idx + i * stride;
        auto amask = sycl::reduce_over_group(
            item_ct1.get_sub_group(),
            (full_mask &
             (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                    idx < size
                ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
                : 0,
            sycl::ext::oneapi::plus<>());
        if (idx < size) {
            /*
            DPCT1084:125: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            function(idx, amask, item_ct1);
        }
    }
}

template<typename F>
inline void blockwise_work_local(int32_t size, F function,
                                 sycl::nd_item<3> item_ct1) {
    for (int32_t i = item_ct1.get_local_id(2); i < size;
         i += item_ct1.get_local_range(2)) {
        /*
        DPCT1084:126: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        function(i);
    }
}

inline int32_t searchtree_entry(int32_t idx) {
    // determine the level by the node index
    // rationale: a complete binary tree with 2^k leaves has 2^k - 1 inner nodes
    int32_t lvl = 31 - sycl::clz(idx + 1);  // == log2(idx + 1)
    int32_t step = searchtree_width >> lvl; // == n / 2^lvl
    int32_t lvl_idx = idx - (1 << lvl) + 1; // index within the level
    return lvl_idx * step + step / 2;
}

} // namespace magma_sampleselect
