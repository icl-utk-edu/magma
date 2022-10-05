/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64

// TODO check precision, as in zlag2c?

void
zclaswp_kernel(
    int n,
    magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA, int ldsa,
    int m, const magma_int_t *ipiv, sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * NB + item_ct1.get_local_id(2);
    int newind;
    magmaFloatComplex res;
    
    if (ind < m) {
        SA   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for (int i=0; i < n; i++) {
            res = MAGMA_C_MAKE( (float)MAGMA_Z_REAL( A[newind+i*lda] ),
                                (float)MAGMA_Z_IMAG( A[newind+i*lda] ));
            SA[i*ldsa] = res; 
        }
    }
}

void
zclaswp_inv_kernel(
    int n,
    magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA, int ldsa,
    int m, const magma_int_t *ipiv, sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * NB + item_ct1.get_local_id(2);
    int newind;
    magmaDoubleComplex res;

    if (ind < m) {
        A    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for (int i=0; i < n; i++) {
            /*
            DPCT1064:248: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            res = MAGMA_Z_MAKE((double)x()(SA[newind + i * ldsa]),
                               (double)y()(SA[newind + i * ldsa]));
            A[i*lda] = res;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    Row i of  A is cast to single precision in row ipiv[i] of SA (incx > 0), or
    row i of SA is cast to double precision in row ipiv[i] of  A (incx < 0),
    for 0 <= i < M.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A.

    @param[in,out]
    A       DOUBLE PRECISION array on the GPU, dimension (LDA,N)
            On entry, the M-by-N matrix to which the row interchanges will be applied.
            TODO update docs

    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in,out]
    SA      REAL array on the GPU, dimension (LDSA,N)
            On exit, the single precision, permuted matrix.
            TODO update docs

    @param[in]
    ldsa    INTEGER.
            LDSA specifies the leading dimension of SA.
        
    @param[in]
    m       The number of rows to be interchanged.

    @param[in]
    ipiv    INTEGER array on the GPU, dimension (M)
            The vector of pivot indices. Row i of A is cast to single 
            precision in row ipiv[i] of SA, for 0 <= i < m. 

    @param[in]
    incx    INTEGER
            If INCX is negative, the pivots are applied in reverse order,
            otherwise in straight-forward order.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_laswp
*******************************************************************************/
extern "C" void
magmablas_zclaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA, magma_int_t ldsa,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue )
{
    int blocks = magma_ceildiv( m, NB );
    sycl::range<3> grid(1, 1, blocks);
    sycl::range<3> threads(1, 1, NB);

    if (incx >= 0)
        /*
        DPCT1049:249: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zclaswp_kernel(n, A, lda, SA, ldsa, m, ipiv,
                                              item_ct1);
                           });
    else
        /*
        DPCT1049:250: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zclaswp_inv_kernel(n, A, lda, SA, ldsa, m, ipiv,
                                                  item_ct1);
                           });
}
