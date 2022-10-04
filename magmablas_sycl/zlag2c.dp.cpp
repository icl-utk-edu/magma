/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

// mixed precision generation has issues with COMPLEX, so use PRECISION_z
#define PRECISION_z

#define BLK_X 64
#define BLK_Y 32

// TODO get rid of global variable!
// Static also doesn't work for HIP
// static __device__ int magma_zlag2c_flag = 0;
dpct::global_memory<int, 0> magma_zlag2c_flag(0);

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    
    Code similar to zlat2c and zlaset.
*/

void zlag2c_kernel(
    int m, int n,
    const magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA,       int ldsa,
    double rmax , sycl::nd_item<3> item_ct1, int *magma_zlag2c_flag)
{
    magmaDoubleComplex tmp;
    double neg_rmax = - rmax;

    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *magma_zlag2c_flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp), MAGMA_Z_IMAG(tmp) );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *magma_zlag2c_flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp), MAGMA_Z_IMAG(tmp) );
            }
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLAG2C converts a double-complex matrix, A,
                 to a single-complex matrix, SA.
    
    RMAX is the overflow for the single-complex arithmetic.
    ZLAG2C checks that all the entries of A are between -RMAX and
    RMAX. If not, the conversion is aborted and a magma_zlag2c_flag is raised.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  m >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the m-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,m).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the m-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,m).
    
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

    @ingroup magma_lag2
*******************************************************************************/
extern "C" void
magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA,       magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info )
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldsa < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    double rmax = (double)lapackf77_slamch("O");

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, magma_ceildiv(n, BLK_Y), magma_ceildiv(m, BLK_X));
    q_ct1.memcpy(magma_zlag2c_flag.get_ptr(), info, sizeof(magma_zlag2c_flag))
        .wait(); // magma_zlag2c_flag = 0

    /*
    DPCT1049:1097: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        magma_zlag2c_flag.init(*((sycl::queue *)(queue->cuda_stream())));

        auto magma_zlag2c_flag_ptr_ct1 = magma_zlag2c_flag.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zlag2c_kernel(m, n, A, lda, SA, ldsa, rmax,
                                           item_ct1, magma_zlag2c_flag_ptr_ct1);
                         });
    });

    q_ct1.memcpy(info, magma_zlag2c_flag.get_ptr(), sizeof(magma_zlag2c_flag))
        .wait(); // info = magma_zlag2c_flag
}
