#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_DEVICE_CUH
#define TRMM_TEMPLATE_DEVICE_CUH

///////////////////////////////////////////////////////////////////////////////////////////////////
// op<trans>( x ) returns x or conj(x).
template<typename T, const int CONJA>
static inline T OP( T& x )
{
    if(CONJA == 1) return conj(x);
    else return x;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trmm modes
lNL: left  - NoTrans - Lower 
lNU: left  - NoTrans - Upper
lTL: left  - Trans   - Lower 
lTU: left  - Trans   - Upper
rNL: right - NoTrans - Lower 
rNU: right - NoTrans - Upper
rTL: right - Trans   - Lower 
rTU: right - Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// lNL, lNU
template<typename T, const int NB>
static 
void trmm_small_template_device_lNx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);

    const int nblocks = magma_ceildiv(n, NB);
    const int nn = (bx < nblocks-1) ? NB : n - (nblocks-1)*NB;
    B += bx * NB * lddb;

    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    /*
    DPCT1064:1422: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    
    // load A and B
    if(ty < m  && tx < m) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < nn && tx < m) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    /*
    DPCT1065:1421: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sA[i * NB + tx] * sB[ty * NB + i];
    rb *= alpha;
    // write B
    if(ty < nn && tx < m) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// lTL, lTU, lCL, lCU
template<typename T, const int NB, const int CONJA>
static 
void trmm_small_template_device_lTx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);

    const int nblocks = magma_ceildiv(n, NB);
    const int nn = (bx < nblocks-1) ? NB : n - (nblocks-1)*NB;
    B += bx * NB * lddb;

    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    /*
    DPCT1065:1423: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // needed because sA will be stored as transposed

    // load A and B
    if(ty < m  && tx < m) sA[tx * NB + ty] = OP<T, CONJA>( A[ty * ldda + tx] );
    if(ty < nn && tx < m) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        /*
        DPCT1064:1426: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    /*
    DPCT1065:1424: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if(uplo == MagmaLower){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    /*
    DPCT1065:1425: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sA[i * NB + tx] * sB[ty * NB + i];
    rb *= alpha;

    // write B
    if(ty < nn && tx < m) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rNL, rNU
template<typename T, const int NB>
static 
void trmm_small_template_device_rNx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);

    const int nblocks = magma_ceildiv(m, NB);
    const int mm = (bx < nblocks-1) ? NB : m - (nblocks-1)*NB;
    B += bx * NB;

    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    /*
    DPCT1064:1428: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    /*
    DPCT1064:1429: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);

    // load A and B
    if(ty < n && tx <  n) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < n && tx < mm) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        /*
        DPCT1064:1430: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        /*
        DPCT1064:1431: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    /*
    DPCT1065:1427: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sB[i * NB + tx] * sA[ty * NB + i];
    rb *= alpha;
    // write B
    if(ty < n && tx < mm) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rTL, rTU, rCL, rCU
template<typename T, const int NB, const int CONJA>
static 
void trmm_small_template_device_rTx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);

    const int nblocks = magma_ceildiv(m, NB);
    const int mm = (bx < nblocks-1) ? NB : m - (nblocks-1)*NB;
    B += bx * NB;

    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    /*
    DPCT1064:1433: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    /*
    DPCT1064:1434: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);

    // load A and B
    if(ty < n && tx < n ) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < n && tx < mm) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        /*
        DPCT1064:1435: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        /*
        DPCT1064:1436: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        /*
        DPCT1064:1437: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    /*
    DPCT1065:1432: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sB[i * NB + tx] * OP<T, CONJA>( sA[i * NB + ty] );
    rb *= alpha;
    // write B
    if(ty < n && tx < mm) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
#endif //TRMM_TEMPLATE_DEVICE_CUH
