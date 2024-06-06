/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256

#define PRECISION_z


// initialize arrays with zero
void
magma_zgpumemzero(  
    magmaDoubleComplex * d, 
    int n, 
    int k ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if (i < n) {
        for (int j = 0; j < k; j++)
            d[i + j * n] = MAGMA_Z_ZERO;
    }
}

// dot product
void
magma_zdot_kernel( 
    int Gs,
    int n, 
    magmaDoubleComplex * v,
    magmaDoubleComplex * r,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    temp[Idx] = (i < n) ? v[i] * r[i] : MAGMA_Z_ZERO;
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            /*
            DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}

// dot product for multiple vectors
void
magma_zblockdot_kernel( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * v,
    magmaDoubleComplex * r,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    // k vectors v(i)
    if (i<n){
        for( j=0; j<k; j++)
            temp[Idx + j * item_ct1.get_local_range(2)] = v[i + j * n] * r[i];
    }
    else {
        for( j=0; j<k; j++)
            temp[Idx + j * item_ct1.get_local_range(2)] = MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<k; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 32 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 16 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 8 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 4 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 2 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 32 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 16 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 8 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 4 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 2 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}

// block reduction for multiple vectors
void
magma_zblockreduce_kernel( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    for( j=0; j<k; j++){
        temp[Idx + j * item_ct1.get_local_range(2)] =
            (i < n) ? vtmp[i + j * n] : MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<k; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:24: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:25: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 32 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 16 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 8 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 4 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 2 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 32 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 16 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 8 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 4 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 2 ];
                temp2[ Idx+j*blockDimX ] += temp2[ Idx+j*blockDimX + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp2[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}

// accelerated reduction for one vector
void
magma_zreduce_kernel_fast( int Gs,
                           int n, 
                           magmaDoubleComplex * vtmp,
                           magmaDoubleComplex * vtmp2 ,
                           sycl::nd_item<3> item_ct1,
                           uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int blockSize = 128;
    int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
    temp[Idx] = MAGMA_Z_ZERO;
    int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
    while (i < Gs ) {
        temp[ Idx  ] += vtmp[ i ];
        temp[Idx] += (i + blockSize < Gs) ? vtmp[i + blockSize]
                                          : MAGMA_Z_ZERO;
        i += gridSize;
    }
    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            /*
            DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:31: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:34: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
        vtmp2[item_ct1.get_group(2)] = temp[0];
    }
}

// accelerated block reduction for multiple vectors
void
magma_zblockreduce_kernel_fast( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int blockSize = 128;
    int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
    int j;

    for( j=0; j<k; j++){
        int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
        temp[Idx + j * (blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ];
            temp[Idx + j * (blockSize)] += (i + (blockSize) < Gs)
                                               ? vtmp[i + j * n + (blockSize)]
                                               : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    /*
    DPCT1065:35: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    /*
    DPCT1065:36: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:38: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:40: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:41: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp2[item_ct1.get_group(2) + j * n] = temp[j * (blockSize)];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of vectors v_i such that

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:44: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = (k) * (local_block_size) * sizeof(magmaDoubleComplex); // k vecs
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        

    if (k>1) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zblockdot_kernel(
                                         Gs[2], n, k, v, r, d1, item_ct1,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    }
    else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zdot_kernel(
                                         Gs[2], n, v, r, d1, item_ct1,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    }
/*
    // not necessary to zero GPU mem
    magma_zgpumemzero<<< Gs, Bs, 0, queue->sycl_stream >>>( d1, n*k,1 );
    magma_zgpumemzero<<< Gs, Bs, 0, queue->sycl_stream >>>( d2, n*k,1 );
    //magmablas_zlaset( MagmaFull, n, k, d1, n, UNKNOWN );
    //magmablas_zlaset( MagmaFull, n, k, d2, n, UNKNOWN );
    while( Gs.x > 1 ) {
        Gs_next.x = magma_ceildiv( Gs.x, Bs.x );
        magma_zblockreduce_kernel<<< Gs_next.x, Bs.x, Ms, queue->sycl_stream >>> 
                                        ( Gs.x, n, k, aux1, aux2 );
        Gs.x = Gs_next.x;
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }
    for( int j=0; j<k; j++) {
            magma_zcopyvector( 1, aux1+j*n, 1, skp+j, 1, UNKNOWN );
    }
*/
   
    if ( k>1) {
        while (Gs[2] > 1) {
            Gs_next[2] = magma_ceildiv(Gs[2], Bs[2]);
            if (Gs_next[2] == 1) Gs_next[2] = 2;
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Ms / 2), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, Gs_next[2] / 2) *
                                              sycl::range<3>(1, 1, Bs[2] / 2),
                                          sycl::range<3>(1, 1, Bs[2] / 2)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zblockreduce_kernel_fast(
                                Gs[2], n, k, aux1, aux2, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            Gs_next[2] = Gs_next[2] / 2;
            Gs[2] = Gs_next[2];
            b = 1 - b;
            if ( b ) { aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }
    }
    else {
        while (Gs[2] > 1) {
            Gs_next[2] = magma_ceildiv(Gs[2], Bs[2]);
            if (Gs_next[2] == 1) Gs_next[2] = 2;
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Ms / 2), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, Gs_next[2] / 2) *
                                              sycl::range<3>(1, 1, Bs[2] / 2),
                                          sycl::range<3>(1, 1, Bs[2] / 2)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zreduce_kernel_fast(
                                Gs[2], n, aux1, aux2, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            Gs_next[2] = Gs_next[2] / 2;
            Gs[2] = Gs_next[2];
            b = 1 - b;
            if ( b ) { aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }
    }

    magma_zcopyvector_async( k, aux1, n, skp, 1, queue );

    return MAGMA_SUCCESS;
}

/**
    Purpose
    -------

    This is an extension of the merged dot product above by chunking
    the set of vectors v_i such that the data always fits into cache.
    It is equivalent to a matrix vecor product Vr where V
    contains few rows and many columns. The computation is the same:

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zgemvmdot(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int rows_left = k;
    int offset = 0;
    int chunk_size = 4;
    // process in chunks of 10 - has to be adapted to hardware and precision
    while( rows_left > (chunk_size) ) {
        magma_zmdotc( n, chunk_size, v+offset*n, r, d1, d2, skp+offset, queue );
        offset = offset + chunk_size;
        rows_left = rows_left-chunk_size;
    }
    // process rest
    magma_zmdotc( n, rows_left, v+offset*n, r, d1, d2, skp+offset, queue ); 


    return MAGMA_SUCCESS;
}
