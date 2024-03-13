#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#ifndef MAGMA_TEMPLATES_H
#define MAGMA_TEMPLATES_H


/******************************************************************************/
template< int n, typename T, typename ID >
void 
magma_getidmax( /*int n,*/ int i, T* x, ID* ind , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:36: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            if (x[i] < x[i + 1024] ||
                (x[i] == x[i + 1024] && ind[i + 1024] < ind[i])) {
                ind[i] = ind[i + 1024]; x[i] = x[i + 1024];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:38: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            if (x[i] < x[i + 512] ||
                (x[i] == x[i + 512] && ind[i + 512] < ind[i])) {
                ind[i] = ind[i + 512]; x[i] = x[i + 512];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            if (x[i] < x[i + 256] ||
                (x[i] == x[i + 256] && ind[i + 256] < ind[i])) {
                ind[i] = ind[i + 256]; x[i] = x[i + 256];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:40: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            if (x[i] < x[i + 128] ||
                (x[i] == x[i + 128] && ind[i + 128] < ind[i])) {
                ind[i] = ind[i + 128]; x[i] = x[i + 128];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:41: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            if (x[i] < x[i + 64] ||
                (x[i] == x[i + 64] && ind[i + 64] < ind[i])) {
                ind[i] = ind[i + 64]; x[i] = x[i + 64];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            if (x[i] < x[i + 32] ||
                (x[i] == x[i + 32] && ind[i + 32] < ind[i])) {
                ind[i] = ind[i + 32]; x[i] = x[i + 32];
            }
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads                                                                       
    // because of implicit warp level synchronization.
    /*
    DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            if (x[i] < x[i + 16] ||
                (x[i] == x[i + 16] && ind[i + 16] < ind[i])) {
                ind[i] = ind[i + 16]; x[i] = x[i + 16];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            if (x[i] < x[i + 8] || (x[i] == x[i + 8] && ind[i + 8] < ind[i])) {
                ind[i] = ind[i + 8]; x[i] = x[i + 8];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            if (x[i] < x[i + 4] || (x[i] == x[i + 4] && ind[i + 4] < ind[i])) {
                ind[i] = ind[i + 4]; x[i] = x[i + 4];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            if (x[i] < x[i + 2] || (x[i] == x[i + 2] && ind[i + 2] < ind[i])) {
                ind[i] = ind[i + 2]; x[i] = x[i + 2];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            if (x[i] < x[i + 1] || (x[i] == x[i + 1] && ind[i + 1] < ind[i])) {
                ind[i] = ind[i + 1]; x[i] = x[i + 1];
            }
        } item_ct1.barrier();
    }
}
// end magma_getidmax


/***************************************************************************//**
    Same as magma_getidmax(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T, typename ID >
void 
magma_getidmax_n( int n, int i, T* x, ID* ind , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            if (x[i] < x[i + 1024] ||
                (x[i] == x[i + 1024] && ind[i + 1024] < ind[i])) {
                ind[i] = ind[i + 1024]; x[i] = x[i + 1024];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            if (x[i] < x[i + 512] ||
                (x[i] == x[i + 512] && ind[i + 512] < ind[i])) {
                ind[i] = ind[i + 512]; x[i] = x[i + 512];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            if (x[i] < x[i + 256] ||
                (x[i] == x[i + 256] && ind[i + 256] < ind[i])) {
                ind[i] = ind[i + 256]; x[i] = x[i + 256];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            if (x[i] < x[i + 128] ||
                (x[i] == x[i + 128] && ind[i + 128] < ind[i])) {
                ind[i] = ind[i + 128]; x[i] = x[i + 128];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            if (x[i] < x[i + 64] ||
                (x[i] == x[i + 64] && ind[i + 64] < ind[i])) {
                ind[i] = ind[i + 64]; x[i] = x[i + 64];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            if (x[i] < x[i + 32] ||
                (x[i] == x[i + 32] && ind[i + 32] < ind[i])) {
                ind[i] = ind[i + 32]; x[i] = x[i + 32];
            }
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads                                              
    // because of implicit warp level synchronization.
    /*
    DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            if (x[i] < x[i + 16] ||
                (x[i] == x[i + 16] && ind[i + 16] < ind[i])) {
                ind[i] = ind[i + 16]; x[i] = x[i + 16];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            if (x[i] < x[i + 8] || (x[i] == x[i + 8] && ind[i + 8] < ind[i])) {
                ind[i] = ind[i + 8]; x[i] = x[i + 8];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            if (x[i] < x[i + 4] || (x[i] == x[i + 4] && ind[i + 4] < ind[i])) {
                ind[i] = ind[i + 4]; x[i] = x[i + 4];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            if (x[i] < x[i + 2] || (x[i] == x[i + 2] && ind[i + 2] < ind[i])) {
                ind[i] = ind[i + 2]; x[i] = x[i + 2];
            }
        } item_ct1.barrier();
    }
    /*
    DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            if (x[i] < x[i + 1] || (x[i] == x[i + 1] && ind[i + 1] < ind[i])) {
                ind[i] = ind[i + 1]; x[i] = x[i + 1];
            }
        } item_ct1.barrier();
    }
}
// end magma_getidmax_n



/***************************************************************************//**
    Does max reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
void
magma_max_reduce( /*int n,*/ int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:60: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:61: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] = max(x[i], x[i + 1024]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:62: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] = max(x[i], x[i + 512]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] = max(x[i], x[i + 256]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] = max(x[i], x[i + 128]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] = max(x[i], x[i + 64]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:66: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] = max(x[i], x[i + 32]);
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:67: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] = max(x[i], x[i + 16]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:68: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] = max(x[i], x[i + 8]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] = max(x[i], x[i + 4]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:70: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] = max(x[i], x[i + 2]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] = max(x[i], x[i + 1]);
        } item_ct1.barrier();
    }
}
// end max_reduce


/***************************************************************************//**
    Same as magma_max_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
void
magma_max_reduce_n( int n, int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] = max(x[i], x[i + 1024]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] = max(x[i], x[i + 512]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:75: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] = max(x[i], x[i + 256]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] = max(x[i], x[i + 128]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:77: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] = max(x[i], x[i + 64]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:78: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] = max(x[i], x[i + 32]);
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:79: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] = max(x[i], x[i + 16]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:80: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] = max(x[i], x[i + 8]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:81: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] = max(x[i], x[i + 4]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:82: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] = max(x[i], x[i + 2]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:83: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] = max(x[i], x[i + 1]);
        } item_ct1.barrier();
    }
}
// end max_reduce_n


/***************************************************************************//**
    max that propogates nan consistently:
    max_nan( 1,   nan ) = nan
    max_nan( nan, 1   ) = nan

    For x=nan, y=1:
    nan < y is false, yields x (nan)

    For x=1, y=nan:
    x < nan    is false, would yield x, but
    isnan(nan) is true, yields y (nan)
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >

inline T max_nan( T x, T y )
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}


/***************************************************************************//**
    Same as magma_max_reduce(), but propogates nan values.

    Does max reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
void
magma_max_nan_reduce( /*int n,*/ int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:84: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:85: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] = max_nan(x[i], x[i + 1024]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:86: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] = max_nan(x[i], x[i + 512]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:87: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] = max_nan(x[i], x[i + 256]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:88: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] = max_nan(x[i], x[i + 128]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:89: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] = max_nan(x[i], x[i + 64]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:90: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] = max_nan(x[i], x[i + 32]);
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:91: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] = max_nan(x[i], x[i + 16]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:92: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] = max_nan(x[i], x[i + 8]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:93: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] = max_nan(x[i], x[i + 4]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:94: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] = max_nan(x[i], x[i + 2]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] = max_nan(x[i], x[i + 1]);
        } item_ct1.barrier();
    }
}
// end max_nan_reduce


/***************************************************************************//**
    Same as magma_max_nan_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
void
magma_max_nan_reduce_n( int n, int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:96: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] = max_nan(x[i], x[i + 1024]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:98: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] = max_nan(x[i], x[i + 512]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:99: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] = max_nan(x[i], x[i + 256]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:100: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] = max_nan(x[i], x[i + 128]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:101: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] = max_nan(x[i], x[i + 64]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:102: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] = max_nan(x[i], x[i + 32]);
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:103: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] = max_nan(x[i], x[i + 16]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] = max_nan(x[i], x[i + 8]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:105: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] = max_nan(x[i], x[i + 4]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:106: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] = max_nan(x[i], x[i + 2]);
        } item_ct1.barrier();
    }
    /*
    DPCT1065:107: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] = max_nan(x[i], x[i + 1]);
        } item_ct1.barrier();
    }
}
// end max_nan_reduce


/***************************************************************************//**
    max reduction, for arbitrary size vector. Leaves max(x) in x[0].
    Uses only one thread block of 512 threads, so is not efficient for really large vectors.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
void
magma_max_nan_kernel( int n, T* x , sycl::nd_item<3> item_ct1, T *smax)
{

    int tx = item_ct1.get_local_id(2);

    smax[tx] = 0;
    for( int i=tx; i < n; i += 512 ) {
        smax[tx] = max_nan( smax[tx], x[i] );
    }
    magma_max_nan_reduce<512>(tx, smax, item_ct1);
    if ( tx == 0 ) {
        x[0] = smax[0];
    }
}


/***************************************************************************//**
    Does sum reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
void
magma_sum_reduce( /*int n,*/ int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:108: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:109: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] += x[i + 1024];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:110: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] += x[i + 512];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:111: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] += x[i + 256];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:112: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] += x[i + 128];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:113: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] += x[i + 64];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:114: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] += x[i + 32];
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:115: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] += x[i + 16];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:116: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] += x[i + 8];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:117: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] += x[i + 4];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:118: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] += x[i + 2];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:119: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] += x[i + 1];
        } item_ct1.barrier();
    }
}
// end sum_reduce


/***************************************************************************//**
    sum reduction, for arbitrary size vector. Leaves sum(x) in x[0].
    Uses only one thread block of 512 threads, so is not efficient for really large vectors.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
void
magma_sum_reduce_kernel( int n, T* x , sycl::nd_item<3> item_ct1, T *sum)
{

    int tx = item_ct1.get_local_id(2);

    sum[tx] = 0;
    for( int i=tx; i < n; i += 512 ) {
        sum[tx] += x[i];
    }
    magma_sum_reduce<512>(tx, sum, item_ct1);
    if ( tx == 0 ) {
        x[0] = sum[0];
    }
}


/***************************************************************************//**
    Same as magma_sum_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
void
magma_sum_reduce_n( int n, int i, T* x , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:120: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:121: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] += x[i + 1024];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:122: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] += x[i + 512];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:123: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] += x[i + 256];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:124: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] += x[i + 128];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:125: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] += x[i + 64];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:126: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] += x[i + 32];
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:127: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] += x[i + 16];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:128: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] += x[i + 8];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] += x[i + 4];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:130: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] += x[i + 2];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:131: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] += x[i + 1];
        } item_ct1.barrier();
    }
}
// end sum_reduce_n


/***************************************************************************//**
    Does sum reduction of each column of M x N array x,
    leaving totals in x[0][j] = sum( x[0:m-1][j] ), for 0 <= j < n.
    Contents of x are destroyed in the process.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int m, int n, typename T >
void
magma_sum_reduce_2d( int i, int j, sycl::local_accessor<T, 2> x,
		     sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:132: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:133: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 1024) {
        if (i < 1024 && i + 1024 < m) {
            x[i][j] += x[i + 1024][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:134: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 512) {
        if (i < 512 && i + 512 < m) {
            x[i][j] += x[i + 512][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:135: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 256) {
        if (i < 256 && i + 256 < m) {
            x[i][j] += x[i + 256][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:136: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 128) {
        if (i < 128 && i + 128 < m) {
            x[i][j] += x[i + 128][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:137: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 64) {
        if (i < 64 && i + 64 < m) {
            x[i][j] += x[i + 64][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:138: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 32) {
        if (i < 32 && i + 32 < m) {
            x[i][j] += x[i + 32][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:139: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 16) {
        if (i < 16 && i + 16 < m) {
            x[i][j] += x[i + 16][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:140: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 8) {
        if (i < 8 && i + 8 < m) {
            x[i][j] += x[i + 8][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:141: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 4) {
        if (i < 4 && i + 4 < m) {
            x[i][j] += x[i + 4][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:142: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 2) {
        if (i < 2 && i + 2 < m) {
            x[i][j] += x[i + 2][j];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:143: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m > 1) {
        if (i < 1 && i + 1 < m) {
            x[i][j] += x[i + 1][j];
        } item_ct1.barrier();
    }
}
// end sum_reduce_2d


/***************************************************************************//**
    Does sum reduction of each "column" of M0 x M1 x M2 array x,
    leaving totals in x[0][j][k] = sum( x[0:m0-1][j][k] ), for 0 <= j < m1, 0 <= k < m2.
    Contents of x are destroyed in the process.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int m0, int m1, int m2, typename T >
void
magma_sum_reduce_3d( int i, int j, int k,
		     sycl::local_accessor<T, 3> x,
                     sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:144: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:145: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 1024) {
        if (i < 1024 && i + 1024 < m0) {
            x[i][j][k] += x[i + 1024][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:146: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 512) {
        if (i < 512 && i + 512 < m0) {
            x[i][j][k] += x[i + 512][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:147: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 256) {
        if (i < 256 && i + 256 < m0) {
            x[i][j][k] += x[i + 256][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:148: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 128) {
        if (i < 128 && i + 128 < m0) {
            x[i][j][k] += x[i + 128][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:149: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 64) {
        if (i < 64 && i + 64 < m0) {
            x[i][j][k] += x[i + 64][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:150: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 32) {
        if (i < 32 && i + 32 < m0) {
            x[i][j][k] += x[i + 32][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:151: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 16) {
        if (i < 16 && i + 16 < m0) {
            x[i][j][k] += x[i + 16][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:152: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 8) {
        if (i < 8 && i + 8 < m0) {
            x[i][j][k] += x[i + 8][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:153: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 4) {
        if (i < 4 && i + 4 < m0) {
            x[i][j][k] += x[i + 4][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:154: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 2) {
        if (i < 2 && i + 2 < m0) {
            x[i][j][k] += x[i + 2][j][k];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:155: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (m0 > 1) {
        if (i < 1 && i + 1 < m0) {
            x[i][j][k] += x[i + 1][j][k];
        } item_ct1.barrier();
    }
}
// end sum_reduce_3d

#endif // MAGMA_TEMPLATES_H
