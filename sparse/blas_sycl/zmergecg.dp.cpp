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

#include <cmath>

#define PRECISION_z
#define COMPLEX
#ifdef COMPLEX
#include <complex>
#endif

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasDoubleComplex hipDoubleComplex
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif


#define BLOCK_SIZE 512


// These routines merge multiple kernels from zmergecg into one
// for a description see 
// "Reformulated Conjugate Gradient for the Energy-Aware 
// Solution of Linear Systems on GPUs (ICPP '13)

// accelerated reduction for one vector
void
magma_zcgreduce_kernel_spmv1( 
    int Gs,
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
    DPCT1065:573: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:574: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:575: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:576: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:577: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:578: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:579: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:580: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ) {
        vtmp2[item_ct1.get_group(2)] = temp[0];
    }
}


// accelerated reduction for two vectors
void
magma_zcgreduce_kernel_spmv2( 
    int Gs,
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
    int j;

    for( j=0; j<2; j++){
        int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
        temp[Idx+j*(blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ];
            temp[Idx + j * (blockSize)] +=
                (i + (blockSize) < Gs)
                    ? vtmp[i + j * n + (blockSize)]
                    /*
                    DPCT1064:583: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    /*
    DPCT1065:581: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    /*
    DPCT1065:582: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:584: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:585: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:586: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:587: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:588: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:589: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
        for( j=0; j<2; j++){
            vtmp2[item_ct1.get_group(2) + j * n] = temp[j * (blockSize)];
        }
    }
}



// computes the SpMV using CSR and the first step of the reduction
void
magma_zcgmerge_spmvcsr_kernel(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    temp[Idx] = MAGMA_Z_ZERO;

    if( i<n ) {
        /*
        DPCT1064:593: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = drowptr[ i ];
        int end = drowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += dval[ j ] * d[ dcolind[j] ];
        z[ i ] =  dot;
        temp[ Idx ] =  d[ i ] * dot;
    }

    /*
    DPCT1065:590: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:591: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:592: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:594: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:595: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:596: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:597: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:598: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:599: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}

// computes the SpMV using ELL and the first step of the reduction
void
magma_zcgmerge_spmvell_kernel(  
    int n,
    int num_cols_per_row,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    temp[Idx] = MAGMA_Z_ZERO;

    if(i < n ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int k = 0; k < num_cols_per_row; k++ ) {
            int col = dcolind [ n * k + i ];
            magmaDoubleComplex val = dval [ n * k + i ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

    /*
    DPCT1065:600: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:601: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:602: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:603: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:604: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:605: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:606: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:607: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:608: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}


// computes the SpMV using ELLPACK and the first step of the reduction
void
magma_zcgmerge_spmvellpack_kernel(  
    int n,
    int num_cols_per_row,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    temp[Idx] = MAGMA_Z_ZERO;

    if(i < n ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int k = 0; k < num_cols_per_row; k++ ) {
            int col = dcolind [ num_cols_per_row * i + k ];
            magmaDoubleComplex val = dval [ num_cols_per_row * i + k ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

    /*
    DPCT1065:609: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:610: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:611: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:612: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:613: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:614: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:615: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:616: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:617: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}


// computes the SpMV using SELL alignment 1 and the first step of the reduction
void
magma_zcgmerge_spmvell_kernelb1(  
    int n,
    int blocksize,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    temp[Idx] = MAGMA_Z_ZERO;

    int idx = item_ct1.get_local_id(2); // local row
    int bdx = item_ct1.get_group(2);    // global block index
    int row = bdx * 256 + idx;  // global row index
    // int lblocksize = ( row + blocksize < num_rows) ? blocksize : ( num_rows - blocksize * (row/blocksize) );
    int lrow = item_ct1.get_local_id(2) % blocksize; // local row;

    if( row < n ) {
        int offset = drowptr[ row/blocksize ];
        int border = (drowptr[ row/blocksize+1 ]-offset)/blocksize;

        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int n = 0; n < border; n++) { 
            int col = dcolind [ offset+ blocksize * n + lrow ];
            magmaDoubleComplex val = dval[ offset+ blocksize * n + lrow ];
            dot = dot + val * d [ col ];
        }
        z[ i ] = dot;
        temp[ Idx ] = d[ i ] * dot;
    }
    
/*
    if(i < n ) {
        int offset = drowptr[ blockIdx.x ];
        int border = (drowptr[ blockIdx.x+1 ]-offset)/blocksize;
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int k = 0; k < border; k++){ 
            int col = dcolind [ offset+ blocksize * k + threadIdx.x ];
            magmaDoubleComplex val = dval[offset+ blocksize * k + threadIdx.x];
            if( val != 0){
                  dot += val*d[col];
            }
        }
        
        
        //magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        //for ( int k = 0; k < num_cols_per_row; k++ ) {
        //    int col = dcolind [ n * k + i ];
        //    magmaDoubleComplex val = dval [ n * k + i ];
        //    if( val != 0)
        //        dot += val * d[ col ];
        //}
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }*/

    /*
    DPCT1065:618: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:619: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:620: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:621: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:622: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:623: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:624: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:625: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:626: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}


// computes the SpMV using ELLRT 8 threads per row
void
magma_zcgmerge_spmvellpackrt_kernel_8(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp,
    magma_int_t T, 
    magma_int_t alignment  ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(i < n ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 4 ) {
            shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }
        }
    }
}

// computes the SpMV using ELLRT 8 threads per row
void
magma_zcgmerge_spmvellpackrt_kernel_16(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp,
    magma_int_t T, 
    magma_int_t alignment  ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(i < n ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 8 ) {
            shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }
        }
    }
}

// computes the SpMV using ELLRT 8 threads per row
void
magma_zcgmerge_spmvellpackrt_kernel_32(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp,
    magma_int_t T, 
    magma_int_t alignment  ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(i < n ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 16 ) {
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }
        }
    }
}


// additional kernel necessary to compute first reduction step
void
magma_zcgmerge_spmvellpackrt_kernel2(  
    int n,
    magmaDoubleComplex * z,
    magmaDoubleComplex * d,
    magmaDoubleComplex * vtmp2 ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    temp[Idx] = (i < n) ? z[i] * d[i] : MAGMA_Z_ZERO;
    /*
    DPCT1065:627: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:628: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:629: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:630: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:631: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:632: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:633: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:634: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:635: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp2[item_ct1.get_group(2)] = temp[0];
    }
}



// computes the SpMV using SELLC
void
magma_zcgmerge_spmvsellc_kernel(   
    int num_rows, 
    int blocksize,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int offset = drowptr[item_ct1.get_group(2)];
    int border = (drowptr[item_ct1.get_group(2) + 1] - offset) / blocksize;

    temp[Idx] = MAGMA_Z_ZERO;

    if(i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int n = 0; n < border; n ++) {
            int col = dcolind [offset+ blocksize * n + Idx ];
            magmaDoubleComplex val = dval[offset+ blocksize * n + Idx];
            if( val != 0) {
                  dot=dot+val*d[col];
            }
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }
    /*
    DPCT1065:636: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:637: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:638: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:639: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:640: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:641: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:642: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:643: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:644: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
void
magma_zcgmerge_spmvsellpt_kernel_8( 
    int num_rows, 
    int blocksize,
    int T,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        /*
        DPCT1065:645: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 4 ) {
            shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:646: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:647: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }
        }
    }
}
// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
void
magma_zcgmerge_spmvsellpt_kernel_16( 
    int num_rows, 
    int blocksize,
    int T,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        /*
        DPCT1065:648: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 8 ) {
            shared[ldx]+=shared[ldx+blocksize*8];
            /*
            DPCT1065:649: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:650: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:651: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
void
magma_zcgmerge_spmvsellpt_kernel_32( 
    int num_rows, 
    int blocksize,
    int T,
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        /*
        DPCT1065:652: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 16 ) {
            shared[ldx]+=shared[ldx+blocksize*16];
            /*
            DPCT1065:653: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];
            /*
            DPCT1065:654: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:655: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:656: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }
        }
    }
}


// kernel to handle scalars
void // rho = beta/tmp; gamma = beta;
magma_zcg_rhokernel(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if( i==0 ) {
        magmaDoubleComplex tmp = skp[1];
        skp[3] = tmp/skp[4];
        skp[2] = tmp;
    }
}

/**
    Purpose
    -------

    Merges the first SpmV using different formats with the dot product 
    and the computation of rho

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix 

    @param[in]
    d1          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    d2          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    dd          magmaDoubleComplex_ptr 
                input vector d

    @param[out]
    dz          magmaDoubleComplex_ptr 
                input vector z

    @param[out]
    skp         magmaDoubleComplex_ptr 
                array for parameters ( skp[3]=rho )

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zcgmerge_spmv1(
    magma_z_matrix A,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(A.num_rows, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    int nthreads_max = queue->sycl_stream()->get_device()
                            .get_info<sycl::info::device::max_work_group_size>();
    /*
    DPCT1083:659: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        

    if ( A.storage_type == Magma_CSR )
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvcsr_kernel(
                                         A.num_rows, A.dval, A.drow, A.dcol, dd,
                                         dz, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    else if ( A.storage_type == Magma_ELLPACKT )
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvellpack_kernel(
                                         A.num_rows, A.max_nnz_row, A.dval,
                                         A.dcol, dd, dz, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    else if ( A.storage_type == Magma_ELL )
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvell_kernel(
                                         A.num_rows, A.max_nnz_row, A.dval,
                                         A.dcol, dd, dz, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    else if ( A.storage_type == Magma_CUCSR ) {
        sycl::queue *cusparseHandle = 0;
        oneapi::mkl::index_base descr;
        magmaDoubleComplex c_one = MAGMA_Z_ONE;
        magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
        cusparseHandle = &dpct::get_default_queue();
        cusparseHandle = queue->sycl_stream();
        descr = oneapi::mkl::index_base::zero;
        /*
        DPCT1026:662: The call to cusparseSetMatType was removed because the
        function call is redundant in DPC++.
        */
        /* //phuong
        descr = oneapi::mkl::index_base::zero;
        cusparseZcsrmv(cusparseHandle, oneapi::mkl::transpose::nontrans,
                       A.num_rows, A.num_cols, A.nnz, (std::complex<double> *)&c_one,
                       descr, (std::complex<double> *)A.dval, A.drow, A.dcol,
                       (std::complex<double> *)dd, (std::complex<double> *)&c_zero,
                       (std::complex<double> *)dz);
                       */
        /*
        DPCT1026:663: The call to cusparseDestroyMatDescr was removed because
        the function call is redundant in DPC++.
        */
        cusparseHandle = nullptr;
        cusparseHandle = 0;
        //descr = 0;
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvellpackrt_kernel2(
                                         A.num_rows, dz, dd, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else if ( A.storage_type == Magma_SELLP && A.alignment == 1 ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvell_kernelb1(
                                         A.num_rows, A.blocksize, A.dval,
                                         A.dcol, A.drow, dd, dz, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else if ( A.storage_type == Magma_SELLP && A.alignment > 1) {
            int num_threadssellp = A.blocksize*A.alignment;
            if ( num_threadssellp > nthreads_max)
              printf("error: too many threads requested (%d) for this device (max %d).\n",
               num_threadssellp, nthreads_max);

            sycl::range<3> block(1, A.alignment, A.blocksize);
            int dimgrid1 = int( sqrt( double( A.numblocks )));
            int dimgrid2 = magma_ceildiv( A.numblocks, dimgrid1 );

            sycl::range<3> gridsellp(1, dimgrid2, dimgrid1);
            /*
            DPCT1083:674: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            int Mssellp = num_threadssellp * sizeof(magmaDoubleComplex);

            if ( A.alignment == 8)
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mssellp), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridsellp * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvsellpt_kernel_8(
                                A.num_rows, A.blocksize, A.alignment, A.dval,
                                A.dcol, A.drow, dd, dz, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });

            else if ( A.alignment == 16)
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mssellp), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridsellp * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvsellpt_kernel_16(
                                A.num_rows, A.blocksize, A.alignment, A.dval,
                                A.dcol, A.drow, dd, dz, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });

            else if ( A.alignment == 32)
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mssellp), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridsellp * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvsellpt_kernel_32(
                                A.num_rows, A.blocksize, A.alignment, A.dval,
                                A.dcol, A.drow, dd, dz, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });

            else
                printf("error: alignment not supported.\n");

        // in case of using SELLP, we can't efficiently merge the 
        // dot product and the first reduction loop into the SpMV kernel
        // as the SpMV grid would result in low occupancy.
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvellpackrt_kernel2(
                                         A.num_rows, dz, dd, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else if ( A.storage_type == Magma_ELLRT ) {
        // in case of using ELLRT, we need a different grid, assigning
        // threads_per_row processors to each row
        // the block size is num_threads
        // fixed values


    int num_blocks = magma_ceildiv( A.num_rows, A.blocksize );

    int num_threads = A.alignment*A.blocksize;

    int real_row_length = magma_roundup( A.max_nnz_row, A.alignment );

    if ( num_threads > nthreads_max)
              printf("error: too many threads requested (%d) for this device (max %d).\n",
               num_threads, nthreads_max);

    int dimgrid1 = int( sqrt( double( num_blocks )));
    int dimgrid2 = magma_ceildiv( num_blocks, dimgrid1 );
    sycl::range<3> gridellrt(1, dimgrid2, dimgrid1);

    /*
    DPCT1083:679: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Mellrt = A.alignment * A.blocksize * sizeof(magmaDoubleComplex);
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads , Ms);

    if ( A.alignment == 32 ) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mellrt), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridellrt *
                                              sycl::range<3>(1, 1, num_threads),
                                          sycl::range<3>(1, 1, num_threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvellpackrt_kernel_32(
                                A.num_rows, A.dval, A.dcol, A.drow, dd, dz, d1,
                                A.alignment, real_row_length, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
    }
    else if ( A.alignment == 16 ) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mellrt), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridellrt *
                                              sycl::range<3>(1, 1, num_threads),
                                          sycl::range<3>(1, 1, num_threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvellpackrt_kernel_16(
                                A.num_rows, A.dval, A.dcol, A.drow, dd, dz, d1,
                                A.alignment, real_row_length, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
    }
    else if ( A.alignment == 8 ) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(Mellrt), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridellrt *
                                              sycl::range<3>(1, 1, num_threads),
                                          sycl::range<3>(1, 1, num_threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zcgmerge_spmvellpackrt_kernel_8(
                                A.num_rows, A.dval, A.dcol, A.drow, dd, dz, d1,
                                A.alignment, real_row_length, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
    }
    else {
        printf("error: alignment %d not supported.\n", int(A.alignment) );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        // in case of using ELLRT, we can't efficiently merge the 
        // dot product and the first reduction loop into the SpMV kernel
        // as the SpMV grid would result in low occupancy.

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zcgmerge_spmvellpackrt_kernel2(
                                         A.num_rows, dz, dd, d1, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }

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
                        magma_zcgreduce_kernel_spmv1(
                            Gs[2], A.num_rows, aux1, aux2, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
        Gs_next[2] = Gs_next[2] / 2;
        Gs[2] = Gs_next[2];
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+4, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_rhokernel(skp, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


/* -------------------------------------------------------------------------- */

// updates x and r and computes the first part of the dot product r*r
void
magma_zcgmerge_xrbeta_kernel(  
    int n, 
    magmaDoubleComplex * x, 
    magmaDoubleComplex * r,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * skp,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    magmaDoubleComplex rho = skp[3];
    /*
    DPCT1064:686: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex mrho = MAGMA_Z_MAKE(-1.0, 0.0) * rho;

    temp[Idx] = MAGMA_Z_ZERO;

    if( i<n ) {
        x[i] += rho * d[i];
        r[i] += mrho * z[i];
        temp[ Idx ] = r[i] * r[i];
    }
    /*
    DPCT1065:683: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ) {
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:684: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ) {
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:685: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ) {
            /*
            DPCT1065:687: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:688: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:689: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:690: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:691: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:692: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 1]; item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ) {
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
        if( Idx < 32 ) {
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ) {
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}

// kernel to handle scalars
void //alpha = beta / gamma
magma_zcg_alphabetakernel(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if( i==0 ) {
        magmaDoubleComplex tmp1 = skp[1];
        skp[0] =  tmp1/skp[2];
        //printf("beta=%e\n", MAGMA_Z_REAL(tmp1));
    }
}

// update search Krylov vector d
void
magma_zcg_d_kernel(  
    int n, 
    magmaDoubleComplex * skp,
    magmaDoubleComplex * r,
    magmaDoubleComplex * d ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex alpha = skp[0];

    if( i<n ) {
        d[i] = r[i] + alpha * d[i];
    }
}



/**
    Purpose
    -------

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    d1          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    d2          magmaDoubleComplex_ptr 
                temporary vector

    @param[in,out]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in,out]
    dr          magmaDoubleComplex_ptr 
                input/output vector r

    @param[in]
    dd          magmaDoubleComplex_ptr 
                input vector d

    @param[in]
    dz          magmaDoubleComplex_ptr 
                input vector z
    @param[in]
    skp         magmaDoubleComplex_ptr 
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zcgmerge_xrbeta(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:694: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 2 * local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zcgmerge_xrbeta_kernel(n, dx, dr, dd, dz, skp, d1,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
            });
    });

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
                        magma_zcgreduce_kernel_spmv1(
                            Gs[2], n, aux1, aux2, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
        Gs_next[2] = Gs_next[2] / 2;
        Gs[2] = Gs_next[2];
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+1, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_alphabetakernel(skp, item_ct1);
                       });

    sycl::range<3> Bs3(1, 1, local_block_size);
    sycl::range<3> Gs3(1, 1, magma_ceildiv(n, local_block_size));
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs3 * Bs3, Bs3),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_d_kernel(n, skp, dr, dd, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

// updates x and r
void
magma_zpcgmerge_xrbeta_kernel(  
    int n, 
    magmaDoubleComplex * x, 
    magmaDoubleComplex * r,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * skp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;

    magmaDoubleComplex rho = skp[3];
    /*
    DPCT1064:698: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex mrho = MAGMA_Z_MAKE(-1.0, 0.0) * rho;

    if( i<n ) {
        x[i] += rho * d[i];
        r[i] += mrho * z[i];
    }
}


// dot product for multiple vectors
void
magma_zmzdotc_one_kernel_1( 
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    // 1 vectors v(i)/w(i)

    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:702: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * w0[i]
            : MAGMA_Z_ZERO;
    temp[Idx + item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:703: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * v0[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:699: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<2; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:700: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:701: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:704: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:705: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:706: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:707: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:708: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:709: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
            vtmp[item_ct1.get_group(2)] = temp[0];
            vtmp[item_ct1.get_group(2) + n] = temp[item_ct1.get_local_range(2)];
    }
}

/**
    Purpose
    -------

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in,out]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in,out]
    dr          magmaDoubleComplex_ptr 
                input/output vector r

    @param[in]
    dd          magmaDoubleComplex_ptr 
                input vector d

    @param[in]
    dz          magmaDoubleComplex_ptr 
                input vector z
    @param[in]
    skp         magmaDoubleComplex_ptr 
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zpcgmerge_xrbeta1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(0), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zpcgmerge_xrbeta_kernel(n, dx, dr, dd, dz, skp, item_ct1,
                                              dpct_local_acc_ct1.get_pointer());
            });
    });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */


/**
    Purpose
    -------

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    d1          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    d2          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    dh          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    dr          magmaDoubleComplex_ptr 
                input/output vector r
                
    @param[in]
    dd          magmaDoubleComplex_ptr 
                input/output vector d

    @param[in]
    skp         magmaDoubleComplex_ptr 
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zpcgmerge_xrbeta2(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dh,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dd, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:712: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 4 * local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zmzdotc_one_kernel_1(n, dr, dh, d1, item_ct1,
                                           dpct_local_acc_ct1.get_pointer());
            });
    });

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
                        magma_zcgreduce_kernel_spmv2(
                            Gs[2], n, aux1, aux2, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
        Gs_next[2] = Gs_next[2] / 2;
        Gs[2] = Gs_next[2];
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+1, 1, queue );
    magma_zcopyvector( 1, aux1+n, 1, skp+6, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_alphabetakernel(skp, item_ct1);
                       });

    sycl::range<3> Bs3(1, 1, local_block_size);
    sycl::range<3> Gs3(1, 1, magma_ceildiv(n, local_block_size));
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs3 * Bs3, Bs3),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_d_kernel(n, skp, dh, dd, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */



// updates x and r
void
magma_zjcgmerge_xrbeta_kernel(  
    int n, 
    magmaDoubleComplex * diag, 
    magmaDoubleComplex * x,     
    magmaDoubleComplex * r,
    magmaDoubleComplex * d,
    magmaDoubleComplex * z,
    magmaDoubleComplex * h,
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * skp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    magmaDoubleComplex rho = skp[3];
    /*
    DPCT1064:720: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex mrho = MAGMA_Z_MAKE(-1.0, 0.0) * rho;

    if( i<n ) {
        x[i] += rho * d[i];
        r[i] += mrho * z[i];
        h[i] = r[i] * diag[i];
    }
    /*
    DPCT1065:716: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:721: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            h[i] * r[i]
            : MAGMA_Z_ZERO;
    temp[Idx + item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:722: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            r[i] * r[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:717: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<2; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:718: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:719: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:723: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:724: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:725: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:726: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:727: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:728: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
            vtmp[item_ct1.get_group(2)] = temp[0];
            vtmp[item_ct1.get_group(2) + n] = temp[item_ct1.get_local_range(2)];
    }
}


/**
    Purpose
    -------

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    d1          magmaDoubleComplex_ptr 
                temporary vector

    @param[in]
    d2          magmaDoubleComplex_ptr 
                temporary vector
                
    @param[in]
    diag        magmaDoubleComplex_ptr 
                inverse diagonal (Jacobi preconditioner)

    @param[in]
    dx          magmaDoubleComplex_ptr
                iteration vector x

    @param[in]
    dr          magmaDoubleComplex_ptr 
                input/output vector r
                
    @param[in]
    dd          magmaDoubleComplex_ptr
                input vector d

                
    @param[in]
    dz          magmaDoubleComplex_ptr
                input vector z
                
    @param[in]
    dh          magmaDoubleComplex_ptr
                input vector h

    @param[in]
    skp         magmaDoubleComplex_ptr 
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zjcgmerge_xrbeta(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr diag,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz,
    magmaDoubleComplex_ptr dh, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:730: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 4 * local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zjcgmerge_xrbeta_kernel(n, diag, dx, dr, dd, dz, dh, d1,
                                              skp, item_ct1,
                                              dpct_local_acc_ct1.get_pointer());
            });
    });

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
                        magma_zcgreduce_kernel_spmv2(
                            Gs[2], n, aux1, aux2, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
        Gs_next[2] = Gs_next[2] / 2;
        Gs[2] = Gs_next[2];
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+1, 1, queue );
    magma_zcopyvector( 1, aux1+n, 1, skp+6, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_alphabetakernel(skp, item_ct1);
                       });

    sycl::range<3> Bs3(1, 1, local_block_size);
    sycl::range<3> Gs3(1, 1, magma_ceildiv(n, local_block_size));
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs3 * Bs3, Bs3),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zcg_d_kernel(n, skp, dh, dd, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */
