/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @precisions normal z -> c d s
   @author Ahmad Abdelfattah
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"

////////////////////////////////////////////////////////////////////////////////
void
extract_diag_sqrt_kernel(int min_mn, magmaDoubleComplex* dA, int ldda, double* dD, int incd,
                         sycl::nd_item<3> item_ct1)
{
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    if( gtx < min_mn ) {
        dD[gtx * incd] = sycl::sqrt(MAGMA_Z_REAL(dA[gtx * ldda + gtx]));
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int DIMX, int DIMY>
void
zscal_shift_hpd_kernel(
        magma_uplo_t uplo, int n,
        magmaDoubleComplex* dA, int ldda,
        double* dD, int incd,
        double miu, double cn, double eps, sycl::nd_item<3> item_ct1,
        magmaDoubleComplex *sD_row, magmaDoubleComplex *sD_col)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);

    const int gbx = item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int gby = item_ct1.get_group(1) * item_ct1.get_local_range(1);

    const int gtx = gbx + tx;
    const int gty = gby + ty;

    magmaDoubleComplex rA = MAGMA_Z_ZERO;
    double rTmp = MAGMA_D_ZERO;
    // read the corresponding segments from diagonal vector
    // for pre-multiplication
    if(ty == 0 && gtx < n) {
        rTmp = dD[gtx * incd];
        /*
        DPCT1064:1: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        sD_row[tx] = MAGMA_Z_DIV(MAGMA_Z_ONE, MAGMA_Z_MAKE(rTmp, 1.));
    }

    // for post multiplication
    const int y_length = min(DIMY, n - gby);
    if( ty == 1 && tx < y_length ) {
        rTmp = dD[ (gby+tx) * incd];
        /*
        DPCT1064:2: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        sD_col[tx] = MAGMA_Z_DIV(MAGMA_Z_ONE, MAGMA_Z_MAKE(rTmp, 1));
    }
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // read
    if(gtx < n && gty < n) {
        rA = dA[gty * ldda + gtx];
    }

    // D^-1 * A -- multiply scale rows
    rA *= sD_row[ tx ];

    // rA * D^-1 -- scale columns
    rA *= sD_col[ ty ];

    /*
    DPCT1064:3: Migrated make_cuDoubleComplex call is used in a macro definition
    and is not valid for all macro uses. Adjust the code.
    */
    rA = (gtx == gty) ? MAGMA_Z_MAKE(1 + (cn * eps), 0.) : rA;
    /*
    DPCT1064:4: Migrated make_cuDoubleComplex call is used in a macro definition
    and is not valid for all macro uses. Adjust the code.
    */
    rA *= MAGMA_Z_MAKE(miu, 0.);

    // write
    if(gtx < n && gty < n) {
        dA[gty * ldda + gtx] = rA;
    }
}

////////////////////////////////////////////////////////////////////////////////
void
dimv_kernel(
        int n,
        magmaDoubleComplex alpha, magmaDoubleComplex *dD, int incd,
                                  magmaDoubleComplex *dx, int incx,
        magmaDoubleComplex beta,  magmaDoubleComplex *dy, int incy,
        bool invert_diagonal, sycl::nd_item<3> item_ct1)
{
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

    magmaDoubleComplex rA = MAGMA_Z_ZERO;
    if (beta != MAGMA_Z_ZERO) {
        if(gtx < n)
            rA = beta * dy[ gtx * incy];
    }

    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;
    if( gtx < n) {
        /*
        DPCT1064:5: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rTmp += (invert_diagonal)
                    ? MAGMA_Z_DIV(MAGMA_Z_ONE, dD[gtx * incd]) * dx[gtx * incx]
                    : dD[gtx * incd] * dx[gtx * incx];
        rTmp *= alpha;
        dy[gtx * incy] = rA + rTmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
// extract the diagonal of an mxn matrix, and write its sqrt to a vector
extern "C"
void
magmablas_zextract_diag_sqrt(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex* dA, magma_int_t ldda,
    double* dD, magma_int_t incd,
    magma_queue_t queue)
{
    const int bx = 256;
    const int min_mn = min(m, n);
    const int nblocks = magma_ceildiv(min_mn, 256);
    sycl::range<3> grid(1, 1, nblocks);
    sycl::range<3> threads(1, 1, bx);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           extract_diag_sqrt_kernel(min_mn, dA, ldda, dD, incd,
                                                    item_ct1);
                       });
}

////////////////////////////////////////////////////////////////////////////////
// two-sided diagonal scaling and shifting for hpd matrices
//  ** A becomes D^-1 * A * D^-1, where D diag( sqrt(a(i,i)) )
//  ** Diagonal elements are also shifted by cn * eps, where cn is a constant
//     of choice and eps is the machine epsilon
//  ** An optional additional scaling by miu is also available
//
// Please see for more details:
//  ** "Exploiting Lower Precision Arithmetic in Solving Symmetric Positive
//      Definite Linear Systems and Least Squares Problems", by Higham et al.
//      http://eprints.maths.manchester.ac.uk/2771/
//
// this kernel ignores uplo and scales the whole matrix
// TODO: scale the upper or the lower triangular part only
extern "C"
void
magmablas_zscal_shift_hpd(
    magma_uplo_t uplo, int n,
    magmaDoubleComplex* dA, int ldda,
    double* dD, int incd,
    double miu, double cn, double eps,
    magma_queue_t queue)
{
    const int DIMX = 32;
    const int DIMY = 8;

    // required check for the kernel to work properly
    assert(DIMX >= DIMY);

    sycl::range<3> threads(1, DIMY, DIMX);
    sycl::range<3> grid(1, magma_ceildiv(n, DIMY), magma_ceildiv(n, DIMX));

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sD_row_acc_ct1(sycl::range<1>(DIMX), cgh);
        sycl::accessor<magmaDoubleComplex, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sD_col_acc_ct1(sycl::range<1>(DIMY), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zscal_shift_hpd_kernel<DIMX, DIMY>(
                                 uplo, n, dA, ldda, dD, incd, miu, cn, eps,
                                 item_ct1, sD_row_acc_ct1.get_pointer(),
                                 sD_col_acc_ct1.get_pointer());
                         });
    });
}

////////////////////////////////////////////////////////////////////////////////
// Computes inverse(diagonal-matrix) x vector
// Diagonal matrix is stored as a dense vector
// operation can be done in-place
extern "C"
void
magmablas_zdimv_invert(
        magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex* dD, magma_int_t incd,
                                  magmaDoubleComplex* dx, magma_int_t incx,
        magmaDoubleComplex beta,  magmaDoubleComplex* dy, magma_int_t incy,
        magma_queue_t queue)
{
    const int nthreads = 256;
    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, nthreads));

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           dimv_kernel(n, alpha, dD, incd, dx, incx, beta, dy,
                                       incy, true, item_ct1);
                       });
}
