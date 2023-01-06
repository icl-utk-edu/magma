/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Stan Tomov
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define PRECISION_z

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8


// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
// for each subtile
//     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
//     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
//     A  += NX
//     AT += NX*ldat
//
// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
//     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
//     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
//
// e.g., with NB=32, NX=16, NY=8 (z precision)
//     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
//     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
//                                                       (AT21 AT22)
void
zgeam_kernel_nn(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, int lda,
    magmaDoubleComplex beta,
    const magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc, sycl::nd_item<3> item_ct1)
{
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int ibx = item_ct1.get_group(2) * NB;
    int iby = item_ct1.get_group(1) * NB;
    int i, j;
    
    A  += ibx + tx + (iby + ty)*lda;
    B  += ibx + tx + (iby + ty)*ldb;
    C  += ibx + tx + (iby + ty)*ldc;
    
    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // perform the operation on NX-by-NB subtile
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    C[j2*ldc] = alpha*A[j2*lda] +beta*B[j2*ldb];
                }
            }
        }
        /*
        DPCT1065:278: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // move to next subtile
        A  += NX;
        B  += NX;
        C  += NX;
    }
}

void
zgeam_kernel_nc(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, int lda,
    magmaDoubleComplex beta,
    const magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc, sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{

    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int ibx = item_ct1.get_group(2) * NB;
    int iby = item_ct1.get_group(1) * NB;
    int i, j;

    A  += iby + tx + (ibx + ty)*lda;
    B  += ibx + tx + (iby + ty)*ldb;
    C  += iby + tx + (ibx + ty)*ldc;

    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from B into sB
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sB[ty + j2][tx] = B[j2*ldb];
                }
            }
        }
        /*
        DPCT1065:279: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // save NB-by-NX subtile from A & sB into C
        i = iby + tx;
        j = ibx + ty + tile*NX;
        #pragma unroll
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
                #pragma unroll
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        C[i2 + j2*ldc] = alpha*A[i2 + j2*lda] + beta*sB[tx + i2][ty + j2];
                    }
                }
            }
        }
        /*
        DPCT1065:280: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // move to next subtile
        A  += NX*lda;
        B  += NX;
        C  += NX*ldc;
    }
}

void
zgeam_kernel_cn(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, int lda,
    magmaDoubleComplex beta,
    const magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc, sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sA)
{

    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int ibx = item_ct1.get_group(2) * NB;
    int iby = item_ct1.get_group(1) * NB;
    int i, j;

    A  += ibx + tx + (iby + ty)*lda;
    B  += iby + tx + (ibx + ty)*ldb;
    C  += iby + tx + (ibx + ty)*ldc;

    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from A into sA
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sA[ty + j2][tx] = A[j2*lda];
                }
            }
        }
        /*
        DPCT1065:281: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // save NB-by-NX subtile from sA & B into C
        i = iby + tx;
        j = ibx + ty + tile*NX;
        #pragma unroll
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
                #pragma unroll
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        C[i2 + j2*ldc] = alpha*sA[tx + i2][ty + j2] + beta*B[i2 + j2*ldb];
                    }
                }
            }
        }
        /*
        DPCT1065:282: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // move to next subtile
        A  += NX;
        B  += NX*ldb;
        C  += NX*ldc;
    }
}

void
zgeam_kernel_cc(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, int lda,
    magmaDoubleComplex beta,
    const magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc, sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sA)
{

    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int ibx = item_ct1.get_group(2) * NB;
    int iby = item_ct1.get_group(1) * NB;
    int i, j;

    A  += ibx + tx + (iby + ty)*lda;
    B  += ibx + tx + (iby + ty)*ldb;
    C  += iby + tx + (ibx + ty)*ldc;

    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from A & B into sA
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sA[ty + j2][tx] = alpha*A[j2*lda] + beta*B[j2*ldb];
                }
            }
        }
        /*
        DPCT1065:283: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // save NB-by-NX subtile from sA into C
        i = iby + tx;
        j = ibx + ty + tile*NX;
        #pragma unroll
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
                #pragma unroll
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        C[i2 + j2*ldc] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
        /*
        DPCT1065:284: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // move to next subtile
        A  += NX;
        B  += NX;
        C  += NX*ldc;
    }
}

/***************************************************************************//**
    Purpose
    -------
    zgeam adds/transposes matrices:
         C = alpha*op( A ) + beta*op( B ).

    The operation supports also the following in-place transformations 
         C = alpha*C + beta*op( B )
         C = alpha*op( A ) + beta*C     

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -      = MagmaNoTrans:   op( A ) = A.
      -      = MagmaTrans:     op( A ) = A**T.
      -      = MagmaConjTrans: op( A ) = A**H.

    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -      = MagmaNoTrans:   op( B ) = B.
      -      = MagmaTrans:     op( B ) = B**T.
      -      = MagmaConjTrans: op( B ) = B**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix op( dA ).  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix op( dA ).  N >= 0.
    
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,k), where k is
            N when transA = MagmaNoTrans,  and is  M  otherwise.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= N 
            when transA = MagmaNoTrans,  and LDDA >= M otherwise.

    @param[in]
    beta    COMPLEX_16
            On entry, BETA specifies the scalar beta.
    
    @param[in]
    dB      COMPLEX_16 array, dimension (LDDB,k), where k is
            N when transB = MagmaNoTrans,  and is  M  otherwise.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= N 
            when transB = MagmaNoTrans,  and LDDB >= M otherwise.
    
    @param[in,out]
    dC      COMPLEX_16 array, dimension (LDDC,N).  
            dC can be input, making the operation in-place, if dC is passed
            as one of the pointers to dA or dB.
            The M-by-N matrix dC.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array dC.  LDDC >= M.  

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_transpose
*******************************************************************************/
extern "C" void
magmablas_zgeam(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( dA == dC && transA != MagmaNoTrans )
        info = -6;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < n )
        info = -7;
    else if ( dB == dC && transB != MagmaNoTrans )
        info = -9;
    else if ( transB == MagmaNoTrans ? lddb < m : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -12;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    sycl::range<3> threads(1, NY, NX);
    sycl::range<3> grid(1, magma_ceildiv(m, NB), magma_ceildiv(n, NB));

    /*
    DPCT1064:285: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) &&
        /*
        DPCT1064:286: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        MAGMA_Z_EQUAL(beta, MAGMA_Z_ZERO))
        // set to 0
        /*
        DPCT1064:287: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmablas_zlaset(MagmaFull, m, n, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dC, lddc,
                         queue);
    else if ( transA == MagmaNoTrans && transB == MagmaNoTrans ){
        sycl::range<3> grid(1, magma_ceildiv(n, NB), magma_ceildiv(m, NB));
        /*
        DPCT1049:288: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgeam_kernel_nn(m, n, alpha, dA, ldda, beta, dB,
                                               lddb, dC, lddc, item_ct1);
                           });
    }
    else if ( transA == MagmaNoTrans )
        /*
        DPCT1049:289: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<2>(NB, NX+1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgeam_kernel_nc(n, m, alpha, dA, ldda,
                                                     beta, dB, lddb, dC, lddc,
                                                     item_ct1, sB_acc_ct1);
                                 });
            });
    else if ( transB == MagmaNoTrans )
        /*
        DPCT1049:290: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(NB, NX+1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgeam_kernel_cn(n, m, alpha, dA, ldda,
                                                     beta, dB, lddb, dC, lddc,
                                                     item_ct1, sA_acc_ct1);
                                 });
            });
    else
        /*
        DPCT1049:291: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(NB, NX+1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgeam_kernel_cc(n, m, alpha, dA, ldda,
                                                     beta, dB, lddb, dC, lddc,
                                                     item_ct1, sA_acc_ct1);
                                 });
            });
}

