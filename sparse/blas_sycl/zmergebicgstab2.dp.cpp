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


// These routines merge multiple kernels from zmergebicgstab into one
// This is the code used for the ASHES2014 paper
// "Accelerating Krylov Subspace Solvers on Graphics Processing Units".
// notice that only CSR format is supported so far.


// accelerated reduction for one vector
void
magma_zreduce_kernel_spmv1(    
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
    DPCT1065:738: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:739: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            /*
            DPCT1065:740: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:741: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:742: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:743: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:744: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:745: Consider replacing sycl::nd_item::barrier() with
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


void
magma_zbicgmerge_spmv1_kernel(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * p,
    magmaDoubleComplex * r,
    magmaDoubleComplex * v,
    magmaDoubleComplex * vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    if( i<n ){
        /*
        DPCT1064:750: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = drowptr[ i ];
        int end = drowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += dval[ j ] * p[ dcolind[j] ];
        v[ i ] =  dot;
    }

    /*
    DPCT1065:746: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    temp[Idx] = (i < n) ? v[i] * r[i] : MAGMA_Z_ZERO;
    /*
    DPCT1065:747: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    /*
    DPCT1065:748: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    /*
    DPCT1065:749: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            /*
            DPCT1065:751: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 32]; item_ct1.barrier();
            /*
            DPCT1065:752: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 16]; item_ct1.barrier();
            /*
            DPCT1065:753: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 8]; item_ct1.barrier();
            /*
            DPCT1065:754: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 4]; item_ct1.barrier();
            /*
            DPCT1065:755: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            temp[Idx] += temp[Idx + 2]; item_ct1.barrier();
            /*
            DPCT1065:756: Consider replacing sycl::nd_item::barrier() with
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

void
magma_zbicgstab_alphakernel(  
                    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1){
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if( i==0 ){
        magmaDoubleComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

/**
    Purpose
    -------

    Merges the first SpmV using CSR with the dot product 
    and the computation of alpha

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                system matrix

    @param[in]
    d1          magmaDoubleComplex_ptr
                temporary vector

    @param[in]
    d2          magmaDoubleComplex_ptr
                temporary vector

    @param[in]
    dp          magmaDoubleComplex_ptr
                input vector p

    @param[in]
    dr          magmaDoubleComplex_ptr
                input vector r

    @param[in]
    dv          magmaDoubleComplex_ptr
                output vector v

    @param[in,out]
    skp         magmaDoubleComplex_ptr
                array for parameters ( skp[0]=alpha )

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge_spmv1(
    magma_z_matrix A,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int n = A.num_rows;
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:759: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        

    if ( A.storage_type == Magma_CSR)
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zbicgmerge_spmv1_kernel(
                                         n, A.dval, A.drow, A.dcol, dp, dr, dv,
                                         d1, item_ct1,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    else
        printf("error: only CSR format supported.\n");

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
                        magma_zreduce_kernel_spmv1(
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


    magma_zcopyvector( 1, aux1, 1, skp, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgstab_alphakernel(skp, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

// accelerated block reduction for multiple vectors
void
magma_zreduce_kernel_spmv2( 
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
    DPCT1065:761: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:762: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                /*
                DPCT1065:763: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                /*
                DPCT1065:764: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                /*
                DPCT1065:765: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                /*
                DPCT1065:766: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                /*
                DPCT1065:767: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                /*
                DPCT1065:768: Consider replacing sycl::nd_item::barrier() with
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

void
magma_zbicgmerge_spmv2_kernel(  
    int n,
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * s,
    magmaDoubleComplex * t,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    if( i<n ){
        /*
        DPCT1064:773: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = drowptr[ i ];
        int end = drowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += dval[ j ] * s[ dcolind[j] ];
        t[ i ] =  dot;
    }

    /*
    DPCT1065:769: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // 2 vectors 
    if (i<n){
            magmaDoubleComplex tmp2 = t[i];
            temp[Idx] = s[i] * tmp2;
            temp[Idx + item_ct1.get_local_range(2)] = tmp2 * tmp2;
    }
    else {
        for( j=0; j<2; j++)
            temp[Idx + j * item_ct1.get_local_range(2)] =
                MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:770: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:771: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:772: Consider replacing sycl::nd_item::barrier() with
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
                DPCT1065:774: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:775: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:776: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:777: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:778: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:779: Consider replacing sycl::nd_item::barrier() with
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
        for( j=0; j<2; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}

void
magma_zbicgstab_omegakernel(  
                    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1){
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

/**
    Purpose
    -------

    Merges the second SpmV using CSR with the dot product 
    and the computation of omega

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
    ds          magmaDoubleComplex_ptr
                input vector s

    @param[in]
    dt          magmaDoubleComplex_ptr
                output vector t

    @param[in,out]
    skp         magmaDoubleComplex_ptr
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge_spmv2(
    magma_z_matrix A,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int n = A.num_rows;
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    int Ms = 2 * local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        
    if ( A.storage_type == Magma_CSR)
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zbicgmerge_spmv2_kernel(
                                         n, A.dval, A.drow, A.dcol, ds, dt, d1,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    else
        printf("error: only CSR format supported.\n");

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
                        magma_zreduce_kernel_spmv2(
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


    magma_zcopyvector( 1, aux1, 1, skp+6, 1, queue );
    magma_zcopyvector( 1, aux1+n, 1, skp+7, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgstab_omegakernel(skp, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

void
magma_zbicgmerge_xrbeta_kernel(  
    int n, 
    magmaDoubleComplex * rr,
    magmaDoubleComplex * r,
    magmaDoubleComplex * p,
    magmaDoubleComplex * s,
    magmaDoubleComplex * t,
    magmaDoubleComplex * x, 
    magmaDoubleComplex * skp,
    magmaDoubleComplex * vtmp ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto temp = (magmaDoubleComplex *)dpct_local;
    int Idx = item_ct1.get_local_id(2);
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + Idx;
    int j;

    auto blockDimX = item_ct1.get_local_range(2);

    magmaDoubleComplex alpha=skp[0];
    magmaDoubleComplex omega=skp[2];

    if( i<n ){
        magmaDoubleComplex sl;
        sl = s[i];
        x[i] = x[i] + alpha * p[i] + omega * sl;
        r[i] = sl - omega * t[i];
    }

    /*
    DPCT1065:784: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // 2 vectors 
    if (i<n){
            magmaDoubleComplex tmp2 = r[i];
            temp[Idx] = rr[i] * tmp2;
            temp[Idx + item_ct1.get_local_range(2)] = tmp2 * tmp2;
    }
    else {
        for( j=0; j<2; j++)
            temp[Idx + j * item_ct1.get_local_range(2)] =
                MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:785: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:786: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:787: Consider replacing sycl::nd_item::barrier() with
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
                DPCT1065:788: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 16];
                /*
                DPCT1065:789: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 8];
                /*
                DPCT1065:790: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 4];
                /*
                DPCT1065:791: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 2];
                /*
                DPCT1065:792: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            for( j=0; j<2; j++)
                temp[Idx + j * item_ct1.get_local_range(2)] +=
                    temp[Idx + j * item_ct1.get_local_range(2) + 1];
                /*
                DPCT1065:793: Consider replacing sycl::nd_item::barrier() with
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
        for( j=0; j<2; j++){
            vtmp[item_ct1.get_group(2) + j * n] =
                temp[j * item_ct1.get_local_range(2)];
        }
    }
}

void
magma_zbicgstab_betakernel(  
    magmaDoubleComplex * skp , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[4]/skp[3];
        magmaDoubleComplex tmp2 = skp[0] / skp[2];
        skp[1] =  tmp1*tmp2;
    }
}

/**
    Purpose
    -------

    Merges the second SpmV using CSR with the dot product 
    and the computation of omega

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
    rr          magmaDoubleComplex_ptr
                input vector rr

    @param[in]
    r           magmaDoubleComplex_ptr
                input/output vector r

    @param[in]
    p           magmaDoubleComplex_ptr
                input vector p

    @param[in]
    s           magmaDoubleComplex_ptr
                input vector s

    @param[in]
    t           magmaDoubleComplex_ptr
                input vector t

    @param[out]
    x           magmaDoubleComplex_ptr
                output vector x

    @param[in]
    skp         magmaDoubleComplex_ptr
                array for parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbicgmerge_xrbeta(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr rr,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    sycl::range<3> Bs(1, 1, local_block_size);
    sycl::range<3> Gs(1, 1, magma_ceildiv(n, local_block_size));
    sycl::range<3> Gs_next(1, 1, 1);
    /*
    DPCT1083:795: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = 2 * local_block_size * sizeof(magmaDoubleComplex);
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

        cgh.parallel_for(sycl::nd_range<3>(Gs * Bs, Bs),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zbicgmerge_xrbeta_kernel(
                                 n, rr, r, p, s, t, x, skp, d1, item_ct1,
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
                        magma_zreduce_kernel_spmv2(
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


    magma_zcopyvector( 1, aux1, 1, skp+4, 1, queue );
    magma_zcopyvector( 1, aux1+n, 1, skp+5, 1, queue );
    sycl::range<3> Bs2(1, 1, 2);
    sycl::range<3> Gs2(1, 1, 1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(Gs2 * Bs2, Bs2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zbicgstab_betakernel(skp, item_ct1);
                       });

    return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */
