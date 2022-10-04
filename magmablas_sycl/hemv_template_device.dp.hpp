#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       Originally based on an implementation by KBLAS
   (https://ecrc.kaust.edu.sa/Pages/Res-kblas.aspx)
*/

#ifndef HEMV_TEMPLATE_DEVICE_CUH
#define HEMV_TEMPLATE_DEVICE_CUH

#define EPT    (NB/TY)
/******************************************************************************/
template <typename T, const int NB, const int TY>
__inline__ void
hemv_diag_device( magma_uplo_t uplo, int N,
                  T alpha, T *A, int ldda,
                           T *X, int incx,
                  T beta , T *Y, int incy , sycl::nd_item<3> item_ct1, T *sA,
                  T *sX)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);
    const int n  = min(NB, N - bx * NB);

    T res = make_FloatingPoint(0.0, 0.0);
    /*
    DPCT1064:864: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    T ry = make_FloatingPoint(0.0, 0.0);

    A += bx * NB * (ldda + 1) + ty * ldda + tx;
    X += bx * NB * incx;
    Y += bx * NB * incy;

    // init sA/sX to zeros
    #pragma unroll
    for(int i = 0; i < NB; i += TY){
        sA[(i + ty) * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    if(ty == 0){
        /*
        DPCT1064:865: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        sX[tx] = make_FloatingPoint(0.0, 0.0);
    }
    if(tx >= n) return;

    // load x/y
    if(ty == 0 && tx < n){
        sX[tx] = X[tx * incx];
        ry = Y[tx * incy] * beta;
    }

    // read sA
    if(n < NB){
        int i;
        #pragma unroll
        for(i = 0; i < n-TY; i+=TY){
            sA[(i+ty) * NB + tx] = A[i * ldda];
        }
        if(ty < (n-i)){
            sA[(i+ty) * NB + tx] = A[i * ldda];
        }
    }else{
        #pragma unroll
        for(int i = 0; i < NB; i+= TY)
            sA[(i + ty) * NB + tx] = A[i * ldda];
    }
    /*
    DPCT1065:859: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // mirror
    if(uplo == MagmaLower){
        #pragma unroll
        for(int i = 0; i < NB; i+=TY){
            if(tx < ty+i){
                sA[(i + ty) * NB + tx] = conj( sA[ tx * NB + (i+ty)] );
            }
        }
    }else{
        #pragma unroll
        for(int i = 0; i < NB; i+=TY){
            if(tx > ty+i){
                sA[(i+ty) * NB + tx] = conj( sA[tx * NB + (i+ty)] );
            }
        }
    }
    /*
    DPCT1065:860: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ignore imaginary part of diagonal elements
    if(ty == 0){
        sA[ tx * NB + tx ] = make_FloatingPoint( real(sA[ tx * NB + tx ]), 0.0 );
    }
    /*
    DPCT1065:861: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // compute
    #pragma unroll
    for(int i = 0; i < NB; i += TY){
        res += sA[ (i + ty) * NB + tx ] * sX[i + ty];
    }

    /*
    DPCT1065:862: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sA[ty * NB + tx] = res;
    /*
    DPCT1065:863: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (ty == 0) {
        res = make_FloatingPoint( 0.0, 0.0 );
        #pragma unroll
        for(int i = 0; i < TY; i++)
            res += sA[i * NB + tx];
        res *= alpha;
        if(tx < n){
            Y[tx * incy] = res + ry;
        }
    }
}

/******************************************************************************/
template <typename T, const int NB, const int TY>
__inline__ void
hemv_lower_device( int N, T alpha,
                   T *A, int ldda,
                   T *X, int incx,
                   T *Y, int incy , sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);
    const int by = item_ct1.get_group(1);
    T *X_, *Y_;
    T rA[EPT], rB[EPT];
    T rv[EPT];
    /*
    DPCT1064:869: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    T rh = make_FloatingPoint(0.0, 0.0);

    const int gridx = magma_ceildiv(N, NB);    // do not use gridDim.x (doesn't work for vbatched)
    const int nfull = (gridx-bx-2); // exclude the diagonal block and the last full/partial block
    const int start = by * (nfull / item_ct1.get_group_range(1)) +
                      min(by, nfull % item_ct1.get_group_range(1));
    const int count = nfull / item_ct1.get_group_range(1) +
                      (by < (nfull % item_ct1.get_group_range(1)));
    if ((bx == gridx - 1) ||
        (by < item_ct1.get_group_range(1) - 1 && count == 0)) return;

    A += bx * NB * (ldda + 1) + start * NB + ty * ldda + tx;
    X += bx * NB * incx;
    X_ = X;
    X += start * NB * incx;
    Y += bx * NB * incy;
    Y_ = Y;
    Y_+= start * NB * incy;

    if(ty == 0){
        sX[tx] = X_[tx * incx];
    }
    /*
    DPCT1065:866: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for(int i = 0; i < EPT; i++){
        /*
        DPCT1064:870: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rv[i] = make_FloatingPoint(0.0, 0.0);
    }

    A  += NB;
    X  += NB * incx;
    Y_ += NB * incy;

    if(count > 0){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rB[k] = A[k * TY * ldda];
        }
    }
    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rA[k] = rB[k];
        }

        A  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rB[k] = A[k * TY * ldda];
            }
        }

        /*
        DPCT1064:873: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
        for(int k = 0; k < EPT; k++){
            rh += rA[k] * sX[k * TY + ty];
            rv[k] += conj( rA[k] ) * X[tx * incx];
        }

        // Horizontal block should be stored in global memory
        /*
        DPCT1065:871: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sA[ty * (NB+1) + tx] = rh;
        /*
        DPCT1065:872: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if(ty == 0)
        {
            /*
            DPCT1064:874: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
            for (int k = 0; k < TY; k++) {
                rh += sA[k * (NB+1) + tx];
            }
            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
        X  += NB * incx;
        Y_ += NB * incy;
    }

    // last irregular block
    const int n = N - (bx+nfull+1)*NB;    // size of remaining full/partial block
    if (by == item_ct1.get_group_range(1) - 1) {
        if(tx < n) {
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rA[k] = A[k * TY * ldda];
            }

            /*
            DPCT1064:877: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
            for(int k = 0; k < EPT; k++){
                rh += rA[k] * sX[k * TY + ty];
                rv[k] += conj( rA[k] ) * X[tx * incx];
            }
        }
        // Horizontal block should be stored in global memory
        /*
        DPCT1065:875: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if(tx < n) {
            sA[ty * (NB+1) + tx] = rh;
        }
        /*
        DPCT1065:876: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if(ty == 0 && tx < n) {
            /*
            DPCT1064:878: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
            for (int k = 0; k < TY; k++) {
                rh += sA[k * (NB+1) + tx];
            }
            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
    }

    /*
    DPCT1065:867: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for(int k = 0; k < EPT; k++){
        sA[(k * TY + ty) * (NB+1) + tx] = rv[k];
    }
    /*
    DPCT1065:868: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(ty == 0){
        /*
        DPCT1064:879: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rv[0] = make_FloatingPoint(0.0, 0.0);
#pragma unroll
        for(int k = 0; k < NB; k++){
            rv[0] += sA[tx * (NB+1) + k];
        }
           rv[0] *= alpha;
           magmablas_atomic_add(&Y[incy * tx], rv[0]);
    }
}

/******************************************************************************/
template <typename T, const int NB, const int TY>
__inline__ void
hemv_upper_device( int N, T alpha,
                   T *A, int ldda,
                   T *X, int incx,
                   T *Y, int incy , sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);
    const int by = item_ct1.get_group(1);
    T *X_, *Y_;
    T rA[EPT], rB[EPT];
    T rv[EPT];
    int addr[EPT];
    T rh = make_FloatingPoint(0.0, 0.0);

    const int gridx = magma_ceildiv(N, NB);    // do not use gridDim.x (doesn't work for vbatched)
    const int nr = N - (gridx-1) * NB;
    const int nblocks = bx;
    const int start = by * (nblocks / item_ct1.get_group_range(1)) +
                      min(by, nblocks % item_ct1.get_group_range(1));
    const int count = nblocks / item_ct1.get_group_range(1) +
                      (by < (nblocks % item_ct1.get_group_range(1)));
    if( bx == 0 || count == 0)return;

    if(bx == gridx-1 && nr < NB)
        A += bx * NB * ldda + start * NB; // + ty * ldda + tx;
    else
        A += bx * NB * ldda + start * NB + ty * ldda + tx;

    X_ = X + bx * NB * incx;
    X += start * NB * incx;
    Y_ = Y + start * NB * incy;
    Y += bx * NB * incy;

    // init
    /*
    DPCT1064:883: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (ty == 0) sX[tx] = make_FloatingPoint(0.0, 0.0);
    if(bx == gridx-1 && nr < NB){
        #pragma unroll
        for(int i = 0; i < EPT; i++){
            /*
            DPCT1064:884: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rv[i] = make_FloatingPoint(0.0, 0.0);
            addr[i] = min(i*TY + ty, nr-1) * ldda + tx;
        }
    }
    else{
        #pragma unroll
        for(int i = 0; i < EPT; i++){
            /*
            DPCT1064:885: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rv[i] = make_FloatingPoint(0.0, 0.0);
            addr[i] = i * TY * ldda;
        }
    }

    if(bx == gridx-1 && nr < NB){
        if(ty == 0 && tx < nr)
            sX[tx] = X_[tx * incx];
    }
    else{
        if(ty == 0)
            sX[tx] = X_[tx * incx];
    }
    /*
    DPCT1065:880: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for(int k = 0; k < EPT; k++)
        rB[k] = A[ addr[k] ];

    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++)
            rA[k] = rB[k];

        A  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++)
                rB[k] = A[ addr[k] ];
        }

        /*
        DPCT1064:888: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
        for(int k = 0; k < EPT; k++){
            rh += rA[k] * sX[k * TY + ty];
            rv[k] += conj( rA[k] ) * X[tx * incx];
        }

        // Horizontal block should be stored in global memory
        /*
        DPCT1065:886: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sA[ty * (NB+1) + tx] = rh;
        /*
        DPCT1065:887: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if(ty == 0)
        {
            /*
            DPCT1064:889: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rh = make_FloatingPoint(0.0, 0.0);
#pragma unroll
            for (int k = 0; k < TY; k++)
                rh += sA[k * (NB+1) + tx];

            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
        X  += NB * incx;
        Y_ += NB * incy;
    }

    /*
    DPCT1065:881: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for(int k = 0; k < EPT; k++){
        sA[(k * TY + ty) * (NB+1) + tx] = rv[k];
    }
    /*
    DPCT1065:882: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(ty == 0){
        rv[0] = make_FloatingPoint(0.0, 0.0);
        #pragma unroll
        for(int k = 0; k < NB; k++){
            rv[0] += sA[tx * (NB+1) + k];
        }
        rv[0] *= alpha;
        if (bx == gridx-1 && nr < NB) {
            if (tx < nr)
                magmablas_atomic_add(&Y[incy * tx], rv[0]);
        }
        else {
            magmablas_atomic_add(&Y[incy * tx], rv[0]);
        }
    }
}

/******************************************************************************/
#endif // HEMV_TEMPLATE_DEVICE_CUH
