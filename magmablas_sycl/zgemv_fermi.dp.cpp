/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"
#include "magma_templates.h"

#define PRECISION_z

#include "gemv_template_device.dp.hpp"

#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v


/******************************************************************************/
// NoTrans kernel
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
zgemvn_template_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy,
    sycl::nd_item<3> item_ct1, magmaDoubleComplex *sdata)
{
    gemvn_template_device<magmaDoubleComplex, DIM_X, DIM_Y, TILE_SIZE>(
        m, n, alpha, A, lda, x, incx, beta, y, incy, item_ct1, sdata);
}


/******************************************************************************/
// Trans/ConjTans kernel
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
void
zgemvc_template_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy,
    sycl::nd_item<3> item_ct1, magmaDoubleComplex *sdata)
{
    gemvc_template_device<magmaDoubleComplex, DIM_X, DIM_Y, TILE_SIZE, trans>(
        m, n, alpha, A, lda, x, incx, beta, y, incy, item_ct1, sdata);
}


/******************************************************************************/
// NoTrans CPU driver
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
zgemvn_template_fermi(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, magma_int_t lda,
    const magmaDoubleComplex * __restrict__ x, magma_int_t incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, magma_int_t incy,
    magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, magma_ceildiv(m, TILE_SIZE));
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    /*
    DPCT1049:363: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<magmaDoubleComplex, 1>
            sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * threads, threads),
            [=](sycl::nd_item<3> item_ct1) {
                zgemvn_template_kernel_fermi<DIM_X, DIM_Y, TILE_SIZE>(
                    m, n, alpha, A, lda, x, incx, beta, y, incy, item_ct1,
                    sdata_acc_ct1.get_pointer());
            });
    });
}


/******************************************************************************/
// Trans/ConjTans CPU driver
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
zgemvc_template_fermi(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, magma_int_t lda,
    const magmaDoubleComplex * __restrict__ x, magma_int_t incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, magma_int_t incy,
    magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, magma_ceildiv(n, TILE_SIZE));
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    if (trans == MagmaConjTrans) {
        /*
        DPCT1049:364: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgemvc_template_kernel_fermi<DIM_X, DIM_Y, TILE_SIZE,
                                                     MagmaConjTrans>(
                            m, n, alpha, A, lda, x, incx, beta, y, incy,
                            item_ct1, sdata_acc_ct1.get_pointer());
                    });
            });
    }
    else {
        /*
        DPCT1049:365: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgemvc_template_kernel_fermi<DIM_X, DIM_Y, TILE_SIZE,
                                                     MagmaTrans>(
                            m, n, alpha, A, lda, x, incx, beta, y, incy,
                            item_ct1, sdata_acc_ct1.get_pointer());
                    });
            });
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX_16 array of dimension ( LDDA, n ) on the GPU.
   
    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      COMPLEX_16 array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
  
    @param[in]
    beta    COMPLEX_16
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      COMPLEX_16 array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv
*******************************************************************************/
extern "C" void
magmablas_zgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy, 
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( trans == MagmaNoTrans ) {
        zgemvn_template_fermi<version(N, 106)>
            ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue );
    }
    else {
        zgemvc_template_fermi<version(T, 189)>
            ( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue );
    }
}
