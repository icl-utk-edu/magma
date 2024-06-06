/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/***************************************************************************//**
    Purpose
    -------
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    This expert routine requires two more arguments than the standard
    zgeqr2, namely, dT and ddA, explained below. The storage for A is
    also not as in the LAPACK's zgeqr2 routine (see below).

    The first is used to output the triangular
    n x n factor T of the block reflector used in the factorization.
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    This version adds internal blocking.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).
    \n
            the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    dtau    COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      COMPLEX_16 array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector
            used in the factorization. The lower triangular part is 0.

    @param[out]
    ddA     COMPLEX_16 array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    @param
    dwork   (workspace) DOUBLE PRECISION array, dimension (3 N)

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v**H

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqr2
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr dwork,
    magma_queue_t queue,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (j_)*(ldda) + (i_))
    #define dT(i_,j_) (dT + (j_)*(k)    + (i_))
    #define BS 32

    magma_int_t i, k;

    magmaDouble_ptr dnorm = (magmaDouble_ptr)dwork;
    magmaDoubleComplex_ptr dwork2 = (magmaDoubleComplex_ptr)(dwork + 2*n);

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(m,n);
    magmablas_dznrm2_cols( m, k, dA(0,0), ldda, dnorm, queue );

    for (magma_int_t b=0; b < k; b += BS) {
        for (i = b; i < min(k, b+BS); ++i) {
            /*   Apply H**H to A(:,i) from the left */
            if (i-b > 0) {
                /* Compute the (i-1)th column of T */
                if ( i-1 > 0 ) {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 1>
                                sum_acc_ct1(sycl::range<1>(BLOCK_SIZE),
                                            cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, 1, i - 1) *
                                        sycl::range<3>(1, 1, BLOCK_SIZE),
                                    sycl::range<3>(1, 1, BLOCK_SIZE)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    magma_zgemv_kernel3(
                                        m - i + 1, dA(i - 1, 0), ldda,
                                        dA(i - 1, i - 1), dwork2, dtau + i - 1,
                                        item_ct1, sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 1>
                                sum_acc_ct1(sycl::range<1>(128), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, 1, i - 1) *
                                        sycl::range<3>(1, 1, i - 1),
                                    sycl::range<3>(1, 1, i - 1)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    magma_ztrmv_kernel2(
                                        dT(0, 0), k, dwork2, dT(0, i - 1),
                                        dtau + i - 1, item_ct1,
                                        sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }

                /* dwork = V**H c */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<magmaDoubleComplex, 1>
                            sum_acc_ct1(sycl::range<1>(BLOCK_SIZE),
                                        cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, i - b) *
                                    sycl::range<3>(1, 1, BLOCK_SIZE),
                                sycl::range<3>(1, 1, BLOCK_SIZE)),
                            [=](sycl::nd_item<3> item_ct1) {
                                magma_zgemv_kernel1(m - b, dA(b, b), ldda,
                                                    dA(b, i), dwork2, item_ct1,
                                                    sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                            });
                    });

                /* dwork = T**H dwork2 */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<magmaDoubleComplex, 1>
                            sum_acc_ct1(sycl::range<1>(128), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, i - b) *
                                                  sycl::range<3>(1, 1, i - b),
                                              sycl::range<3>(1, 1, i - b)),
                            [=](sycl::nd_item<3> item_ct1) {
                                magma_ztrmv_tkernel(dT(b, b), k, dwork2,
                                                    dwork2 + i - b, item_ct1,
                                                    sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                            });
                    });

                /* c = c - V dwork2 */
                if ( m-b > 0 ) {
                    sycl::range<3> blocks3(1, 1,
                                           magma_ceildiv(m - b, BLOCK_SIZE));
                    sycl::range<3> threads3(1, 1, BLOCK_SIZE);
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(blocks3 * threads3, threads3),
                            [=](sycl::nd_item<3> item_ct1) {
                                magma_zgemv_kernel2(m - b, i - b, dA(b, b),
                                                    ldda, dwork2 + i - b,
                                                    dA(b, i), item_ct1);
                            });
                }
            }

            /*   Adjust the dnorm[i] to hold the norm of A(i:m,i) */
            if ( i > 0 ) {
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<double, 1>
                            sum_acc_ct1(sycl::range<1>(BLOCK_SIZE),
                                        cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, i),
                                              sycl::range<3>(1, 1, i)),
                            [=](sycl::nd_item<3> item_ct1) {
                                magma_dznrm2_adjust_kernel(
                                    dnorm + i, dA(0, i), item_ct1,
                                    sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                            });
                    });
            }

            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i)
                1. 1 is not yet put on the diagonal of A
                2. Elements above the diagonal are copied in ddA and
                   the ones in A are set to zero
                3. update T */
            magma_zlarfgx_gpu( m-i, dA(i, i), dA(min(i+1,m),i), dtau+i,
                               dnorm+i, ddA + i + i*n, i, queue );

            if (i == 0) {
                /*
                DPCT1064:442: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                magmaDoubleComplex tt = MAGMA_Z_ONE;
                magmablas_zlacpy( MagmaFull, 1, 1, dtau, 1, dT(0,0), 1, queue );
                magma_zsetmatrix_async(1, 1, &tt, 1, dA(i, i), 1, queue );
            }
        }
        if ( i-1 > 0 ) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, i - 1) *
                                              sycl::range<3>(1, 1, BLOCK_SIZE),
                                          sycl::range<3>(1, 1, BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zgemv_kernel3(m - i + 1, dA(i - 1, 0), ldda,
                                                dA(i - 1, i - 1), dwork2,
                                                dtau + i - 1, item_ct1,
                                                sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        sum_acc_ct1(sycl::range<1>(128), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, i - 1) *
                                              sycl::range<3>(1, 1, i - 1),
                                          sycl::range<3>(1, 1, i - 1)),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_ztrmv_kernel2(
                                dT(0, 0), k, dwork2, dT(0, i - 1), dtau + i - 1,
                                item_ct1, sum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        }

        /* Apply the transformations to the trailing matrix. */
        //magma_zlarfb2_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
        magma_zlarfb2_gpu(
                           m-b, k-i, BS,
                           dA(b, b), ldda, dT+b+b*k, k,
                           dA(b, i), ldda, dwork2, k-i, queue );
    }

    return *info;
} /* magma_zgeqr2 */
