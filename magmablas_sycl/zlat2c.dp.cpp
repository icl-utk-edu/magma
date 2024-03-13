/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
       @author Mark Gates
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

// mixed precision generation has issues with COMPLEX, so use PRECISION_z
#define PRECISION_z

#define BLK_X 64
#define BLK_Y 32

// TODO get rid of global variable!
// Static also doesn't work for HIP   
//static __device__ int magma_zlat2c_flag = 0;
dpct::global_memory<int, 0> magma_zlat2c_flag(0);

/*
    Divides matrix into ceil( n/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    Updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.
    
    Code similar to zlag2c and zlaset.
*/

void zlat2c_lower(
    int n,
    const magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA,       int ldsa,
    double rmax , sycl::nd_item<3> item_ct1, int *magma_zlat2c_flag)
{
    magmaDoubleComplex tmp;
    double neg_rmax = - rmax;

    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < n && ind + BLK_X > iby ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *magma_zlat2c_flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp),
                                           MAGMA_Z_IMAG(tmp) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *magma_zlat2c_flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp),
                                           MAGMA_Z_IMAG(tmp) );
            }
        }
    }
}


/*
    Similar to zlat2c_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.
    
    Code similar to zlag2c and zlaset.
*/

void zlat2c_upper(
    int n,
    const magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA,       int ldsa,
    double rmax , sycl::nd_item<3> item_ct1, int *magma_zlat2c_flag)
{
    magmaDoubleComplex tmp;
    double neg_rmax = - rmax;

    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < n && ind < iby + BLK_Y ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *magma_zlat2c_flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp),
                                           MAGMA_Z_IMAG(tmp) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    tmp = A[j*lda];
                    if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                         #if defined(PRECISION_z) || defined(PRECISION_c)
                         || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                         #endif
                        )
                    {
                        *magma_zlat2c_flag = 1;
                    }
                    SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp),
                                               MAGMA_Z_IMAG(tmp) );
                }
            }
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLAT2C converts a double-complex matrix, A,
                 to a single-complex matrix, SA.
    
    RMAX is the overflow for the single-complex arithmetic.
    ZLAT2C checks that all the entries of A are between -RMAX and
    RMAX. If not, the conversion is aborted and a magma_zlat2c_flag is raised.
        
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix A to be converted.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the n-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,n).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the n-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,n).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA on exit is unspecified.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lat2
*******************************************************************************/
extern "C" void
magmablas_zlat2c(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr  A, magma_int_t lda,
    magmaFloatComplex_ptr        SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info )
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    *info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,n) )
        *info = -4;
    else if ( ldsa < max(1,n) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( n == 0 ) {
        return;
    }
    
    double rmax = (double)lapackf77_slamch("O");

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, magma_ceildiv(n, BLK_Y), magma_ceildiv(n, BLK_X));
    q_ct1.memcpy(magma_zlat2c_flag.get_ptr(), info, sizeof(magma_zlat2c_flag))
        .wait(); // magma_zlat2c_flag = 0

    if (uplo == MagmaLower) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                magma_zlat2c_flag.init(
                    *((sycl::queue *)(queue->sycl_stream())));

                auto magma_zlat2c_flag_ptr_ct1 = magma_zlat2c_flag.get_ptr();

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlat2c_lower(n, A, lda, SA, ldsa, rmax,
                                                  item_ct1,
                                                  magma_zlat2c_flag_ptr_ct1);
                                 });
            });
    }
    else if (uplo == MagmaUpper) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                magma_zlat2c_flag.init(
                    *((sycl::queue *)(queue->sycl_stream())));

                auto magma_zlat2c_flag_ptr_ct1 = magma_zlat2c_flag.get_ptr();

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlat2c_upper(n, A, lda, SA, ldsa, rmax,
                                                  item_ct1,
                                                  magma_zlat2c_flag_ptr_ct1);
                                 });
            });
    }

    q_ct1.memcpy(info, magma_zlat2c_flag.get_ptr(), sizeof(magma_zlat2c_flag))
        .wait(); // info = magma_zlat2c_flag
}
