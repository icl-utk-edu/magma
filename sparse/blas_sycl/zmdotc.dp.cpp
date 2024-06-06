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

#define COMPLEX



// dot product for multiple vectors
void
magma_zmdotc1_kernel_1( 
    int Gs,
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

    auto blockDimX = item_ct1.get_local_range(2);

    // 1 vectors v(i)/w(i)

    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:236: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * w0[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:233: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
            temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:234: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
            temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:235: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
                temp[ Idx ] += temp[ Idx + 32 ];
                /*
                DPCT1065:237: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 16 ];
                /*
                DPCT1065:238: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 8 ];
                /*
                DPCT1065:239: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 4 ];
                /*
                DPCT1065:240: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 2 ];
                /*
                DPCT1065:241: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 1 ];
                /*
                DPCT1065:242: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
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
    
    if ( Idx == 0 ){
            vtmp[item_ct1.get_group(2)] = temp[0];
    }
}



// block reduction for 1 vectors
void
magma_zmdotc1_kernel_2( 
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

        int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
        /*
        DPCT1064:245: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        temp[Idx] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx  ] += vtmp[ i ];
            temp[Idx] += (i + (blockSize) < Gs)
                             ? vtmp[i + (blockSize)]
                             /*
                             DPCT1064:246: Migrated make_cuDoubleComplex call is
                             used in a macro definition and is not valid for all
                             macro uses. Adjust the code.
                             */
                             : MAGMA_Z_ZERO;
            i += gridSize;
        }
    /*
    DPCT1065:243: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
            temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:244: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
                temp[ Idx ] += temp[ Idx + 32 ];
                /*
                DPCT1065:247: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 16 ];
                /*
                DPCT1065:248: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 8 ];
                /*
                DPCT1065:249: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 4 ];
                /*
                DPCT1065:250: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 2 ];
                /*
                DPCT1065:251: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                temp[ Idx ] += temp[ Idx + 1 ];
                /*
                DPCT1065:252: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
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
    
    if ( Idx == 0 ){
            vtmp2[item_ct1.get_group(2)] = temp[0];
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 1 vectors such that

    skp[0] = [ <v_0,w_0> ]

    Returns the vector skp.
    In case there are less dot products required, an easy workaround is
    given by doubling input.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and w_i

    @param[in]
    v0          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w0          magmaDoubleComplex_ptr                 
                input vector                           

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[4] of scalar products [<v_i, w_i>]
                This vector is located on the host

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc1(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
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
    DPCT1083:254: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = (local_block_size) * sizeof(magmaDoubleComplex); // 1 skp
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zmdotc1_kernel_1(Gs[2], n, v0, w0, d1, item_ct1,
                                       dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
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
                        magma_zmdotc1_kernel_2(
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
    
        // copy vectors to host
    magma_zgetvector( 1 , aux1, 1, skp, 1, queue );
    

    return MAGMA_SUCCESS;
}

//        2 dot products     //


// initialize arrays with zero
void
magma_zmdotc2_gpumemzero(  
    magmaDoubleComplex * d, 
    int n ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if (i < n) {
        for (int j = 0; j < 2; j++)
            d[i + j * n] = MAGMA_Z_ZERO;
    }
}


// dot product for multiple vectors
void
magma_zmdotc2_kernel_1( 
    int Gs,
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * v1,
    magmaDoubleComplex * w1,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    // 2 vectors v(i)/w(i)

    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:259: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * w0[i]
            : MAGMA_Z_ZERO;

    temp[Idx + item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:260: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v1[i] * w1[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:256: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:257: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:258: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:261: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:262: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:263: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:264: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:265: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:266: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
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
    
    if ( Idx == 0 ){
        for( j=0; j<2; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}



// block reduction for 2 vectors
void
magma_zmdotc2_kernel_2( 
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
        /*
        DPCT1064:269: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        temp[Idx + j * (blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ];
            temp[Idx + j * (blockSize)] +=
                (i + (blockSize) < Gs)
                    ? vtmp[i + j * n + (blockSize)]
                    /*
                    DPCT1064:270: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    /*
    DPCT1065:267: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:268: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:271: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:272: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:273: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:274: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:275: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:276: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
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

    if ( Idx == 0 ){
        for( j=0; j<2; j++){
            vtmp2[item_ct1.get_group(2) + j * n] = temp[j * (blockSize)];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 2 vectors such that

    skp[0,1,2,3] = [ <v_0,w_0>, <v_1,w_1> ]

    Returns the vector skp.
    In case there are less dot products required, an easy workaround is
    given by doubling input.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and w_i

    @param[in]
    v0          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w0          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    v1          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    w1          magmaDoubleComplex_ptr                 
                input vector                             

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[3] of scalar products [<v_i, w_i>]
                This vector is located on the host

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc2(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
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
    DPCT1083:278: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 2 * (local_block_size) * sizeof(magmaDoubleComplex); // 4 skp
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zmdotc2_kernel_1(Gs[2], n, v0, w0, v1, w1, d1, item_ct1,
                                       dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
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
                        magma_zmdotc2_kernel_2(
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
    
        // copy vectors to host
    magma_zgetvector( 2 , aux1, n, skp, 1, queue );
    

    return MAGMA_SUCCESS;
}


//        3 dot products     //


// initialize arrays with zero
void
magma_zmdotc3_gpumemzero(  
    magmaDoubleComplex * d, 
    int n ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if (i < n) {
        for (int j = 0; j < 3; j++)
            d[i + j * n] = MAGMA_Z_ZERO;
    }
}


// dot product for multiple vectors
void
magma_zmdotc3_kernel_1( 
    int Gs,
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * v1,
    magmaDoubleComplex * w1,
    magmaDoubleComplex * v2,
    magmaDoubleComplex * w2,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    // 3 vectors v(i)/w(i)

    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:283: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * w0[i]
            : MAGMA_Z_ZERO;

    temp[Idx + item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:284: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v1[i] * w1[i]
            : MAGMA_Z_ZERO;

    temp[Idx + 2 * item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:285: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v2[i] * w2[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:280: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<3; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:281: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<3; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:282: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:286: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:287: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:288: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:289: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:290: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:291: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<3; j++){
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
        for( j=0; j<3; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}



// block reduction for 3 vectors
void
magma_zmdotc3_kernel_2( 
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

    for( j=0; j<3; j++){
        int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
        /*
        DPCT1064:294: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        temp[Idx + j * (blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ];
            temp[Idx + j * (blockSize)] +=
                (i + (blockSize) < Gs)
                    ? vtmp[i + j * n + (blockSize)]
                    /*
                    DPCT1064:295: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    /*
    DPCT1065:292: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<3; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    /*
    DPCT1065:293: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:296: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:297: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:298: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:299: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:300: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<3; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:301: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<3; j++){
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
        for( j=0; j<3; j++){
            vtmp2[item_ct1.get_group(2) + j * n] = temp[j * (blockSize)];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 4 vectors such that

    skp[0,1,2,3] = [ <v_0,w_0>, <v_1,w_1>, <v_2,w_2>, <v3,w_3> ]

    Returns the vector skp.
    In case there are less dot products required, an easy workaround is
    given by doubling input.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and w_i

    @param[in]
    v0          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w0          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    v1          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    w1          magmaDoubleComplex_ptr                 
                input vector          

    @param[in]
    v2          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w2          magmaDoubleComplex_ptr                 
                input vector                           

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[3] of scalar products [<v_i, w_i>]
                This vector is located on the host

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc3(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr v2, 
    magmaDoubleComplex_ptr w2,
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
    DPCT1083:303: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 3 * (local_block_size) * sizeof(magmaDoubleComplex); // 4 skp
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        
    // magma_zmdotc3_gpumemzero<<< Gs, Bs, 0, queue->sycl_stream() >>>( d1, n );

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zmdotc3_kernel_1(Gs[2], n, v0, w0, v1, w1, v2, w2, d1,
                                       item_ct1,
                                       dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
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
                        magma_zmdotc3_kernel_2(
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
    
        // copy vectors to host
    magma_zgetvector( 3 , aux1, n, skp, 1, queue );

    return MAGMA_SUCCESS;
}



//      4 dot products //


// initialize arrays with zero
void
magma_zmdotc4_gpumemzero(  
    magmaDoubleComplex * d, 
    int n ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if (i < n) {
        for (int j = 0; j < 4; j++)
            d[i + j * n] = MAGMA_Z_ZERO;
    }
}


// dot product for multiple vectors
void
magma_zmdotc4_kernel_1( 
    int Gs,
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * v1,
    magmaDoubleComplex * w1,
    magmaDoubleComplex * v2,
    magmaDoubleComplex * w2,
    magmaDoubleComplex * v3,
    magmaDoubleComplex * w3,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    // 4 vectors v(i)/w(i)

    temp[Idx] =
        (i < n)
            ?
            /*
            DPCT1064:308: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v0[i] * w0[i]
            : MAGMA_Z_ZERO;

    temp[Idx + item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:309: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v1[i] * w1[i]
            : MAGMA_Z_ZERO;

    temp[Idx + 2 * item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:310: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v2[i] * w2[i]
            : MAGMA_Z_ZERO;

    temp[Idx + 3 * item_ct1.get_local_range(2)] =
        (i < n)
            ?
            /*
            DPCT1064:311: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            v3[i] * w3[i]
            : MAGMA_Z_ZERO;

    /*
    DPCT1065:305: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        for( j=0; j<4; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 128];
        }
    }
    /*
    DPCT1065:306: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<4; j++){
            temp[Idx + j * item_ct1.get_local_range(2)] +=
                temp[Idx + j * item_ct1.get_local_range(2) + 64];
        }
    }
    /*
    DPCT1065:307: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 32];
                /*
                DPCT1065:312: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:313: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:314: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:315: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:316: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:317: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<4; j++){
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
        for( j=0; j<4; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}



// block reduction for 4 vectors
void
magma_zmdotc4_kernel_2( 
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

    for( j=0; j<4; j++){
        int i = item_ct1.get_group(2) * (blockSize * 2) + Idx;
        temp[Idx+j*(blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ];
            temp[Idx + j * (blockSize)] +=
                (i + (blockSize) < Gs)
                    ? vtmp[i + j * n + (blockSize)]
                    /*
                    DPCT1064:320: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    /*
    DPCT1065:318: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        for( j=0; j<4; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    /*
    DPCT1065:319: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef COMPLEX
        if( Idx < 32 ){
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:321: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:322: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:323: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:324: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:325: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:326: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }
    #endif
    #ifdef REAL
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<4; j++){
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
        for( j=0; j<4; j++){
            vtmp2[item_ct1.get_group(2) + j * n] = temp[j * (blockSize)];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 4 vectors such that

    skp[0,1,2,3] = [ <v_0,w_0>, <v_1,w_1>, <v_2,w_2>, <v3,w_3> ]

    Returns the vector skp.
    In case there are less dot products required, an easy workaround is
    given by doubling input.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and w_i

    @param[in]
    v0          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w0          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    v1          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    w1          magmaDoubleComplex_ptr                 
                input vector          

    @param[in]
    v2          magmaDoubleComplex_ptr     
                input vector               

    @param[in]
    w2          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    v3          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]
    w3          magmaDoubleComplex_ptr                 
                input vector          

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[4] of scalar products [<v_i, w_i>]
                This vector is located on the host

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc4(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr v2, 
    magmaDoubleComplex_ptr w2,
    magmaDoubleComplex_ptr v3, 
    magmaDoubleComplex_ptr w3,
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
    DPCT1083:328: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 4 * (local_block_size) * sizeof(magmaDoubleComplex); // 4 skp
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(Gs * Bs, Bs), [=](sycl::nd_item<3> item_ct1) {
                magma_zmdotc4_kernel_1(Gs[2], n, v0, w0, v1, w1, v2, w2, v3, w3,
                                       d1, item_ct1,
                                       dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
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
                        magma_zmdotc4_kernel_2(
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
    
        // copy vectors to host
    magma_zgetvector( 4 , aux1, n, skp, 1, queue );
    
    return MAGMA_SUCCESS;
}
