/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       [zcds]gemm_fermi.cu          defines the CPU driver.
       [zcds]gemm_fermi_kernels.h   defines the block sizes for each precision.
       gemm_stencil_defs.h          defines types and functions for precision-independent code.
       
       These files are included multiple times, once for each transpose version.
       gemm_stencil.cuh             defines the GPU kernel (device function).
       gemm_kernel.cuh              defines the GPU kernel (global function).
       
       The batched version uses gemm_kernel_batched.cuh instead of gemm_kernel.cuh.
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"

#define PRECISION_z

#include "zgemm_fermi_kernels.h"

/***************************************************************************//**
    Purpose
    -------
    ZGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
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
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( dA )  and of the  matrix dC.  M  must  be at least  zero.
    
    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( dB ) and the number of columns of the matrix dC. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( dA ) and the number of rows of the matrix op( dB ). K must
            be at least  zero.
    
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( LDA, ka ), where ka is
            k  when  transA = MagmaNoTrans,  and is  m  otherwise.
            Before entry with  transA = MagmaNoTrans,  the leading  m by k
            part of the array dA must contain the matrix dA, otherwise
            the leading  k by m  part of the array dA must contain  the
            matrix dA.
    
    @param[in]
    ldda    INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  transA = MagmaNoTrans then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).
    
    @param[in]
    dB      COMPLEX_16 array of DIMENSION ( LDB, kb ), where kb is
            n  when  transB = MagmaNoTrans,  and is  k  otherwise.
            Before entry with  transB = MagmaNoTrans,  the leading  k by n
            part of the array dB must contain the matrix dB, otherwise
            the leading  n by k  part of the array dB must contain  the
            matrix dB.
    
    @param[in]
    lddb    INTEGER.
            On entry, LDB specifies the first dimension of dB as declared
            in the calling (sub) program. When  transB = MagmaNoTrans then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).
    
    @param[in]
    beta    COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.
    
    @param[in,out]
    dC      COMPLEX_16 array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  dC must
            contain the matrix  dC,  except when  beta  is zero, in which
            case dC need not be set on entry.
            On exit, the array  dC  is overwritten by the  m by n  matrix
            ( alpha*op( dA )*op( dB ) + beta*dC ).
    
    @param[in]
    lddc    INTEGER.
            On entry, LDC specifies the first dimension of dC as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm
*******************************************************************************/
extern "C" void
magmablas_zgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
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
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;
    
    size_t offsetA = 0;
    size_t offsetB = 0;

    int TransA = 2, TransB = 2;
    if      ( transA == MagmaTrans )
        TransA = 1;
    else if ( transA == MagmaNoTrans )
        TransA = 0;
    
    if      ( transB == MagmaTrans )
        TransB = 1;
    else if ( transB == MagmaNoTrans )
        TransB = 0;

    magma_int_t Am = ( ! TransA ? m : k);
    magma_int_t An = (!TransA ? k : m);
    magma_int_t Bm = ( ! TransB ? k : n);
    magma_int_t Bn = (!TransB ? n : k);
    size_t sizeA = (size_t) ldda * (An - 1) + Am;
    size_t sizeB = (size_t) lddb * (Bn - 1) + Bm;

    size_t CUBLAS_MAX_1DBUF_SIZE = ((1 << 27) - 512);
    if ( sizeA >= CUBLAS_MAX_1DBUF_SIZE ||
         sizeB >= CUBLAS_MAX_1DBUF_SIZE )
    {
        magma_zgemm( transA, transB, m, n, k, alpha,
                     dA, ldda, dB, lddb,
                     beta, dC, lddc, queue );
        return;
    }

    // Set up grids
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);

    offsetA = offsetA/sizeof(dA[0]);
    offsetB = offsetB/sizeof(dB[0]);
 
    if ( TransA == 0 && TransB == 0 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_nn),
                               magma_ceildiv(m, BLK_M_nn));
        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_nn(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 0 && TransB == 1 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_nt),
                               magma_ceildiv(m, BLK_M_nt));
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_nt(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 0 && TransB == 2 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_nc),
                               magma_ceildiv(m, BLK_M_nc));
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_nc(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 1 && TransB == 0 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_tn),
                               magma_ceildiv(m, BLK_M_tn));
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_tn(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 1 && TransB == 1 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_tt),
                               magma_ceildiv(m, BLK_M_tt));
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_tt(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 1 && TransB == 2 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_tc),
                               magma_ceildiv(m, BLK_M_tc));
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_tc(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 2 && TransB == 0 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_cn),
                               magma_ceildiv(m, BLK_M_cn));
        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_cn(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 2 && TransB == 1 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_ct),
                               magma_ceildiv(m, BLK_M_ct));
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_ct(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }
    else if ( TransA == 2 && TransB == 2 ) {
        sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N_cc),
                               magma_ceildiv(m, BLK_M_cc));
        /*
        DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
  ((sycl::queue *)(queue->cuda_stream()))
      ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                      zgemm_kernel_fermi_cc(m, n, k, dA, ldda, dB, lddb, dC,
                                            lddc, alpha, beta, (int)offsetA,
                                            (int)offsetB);
                     });
    }

}
