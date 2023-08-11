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
#include "magmasparse_internal.h"

#define blksize 512


// TODO get rid of global variable!
dpct::global_memory<int, 0> flag(0);

void
magmaint_clag2z_sparse(  int M, int N,
                  const magmaFloatComplex *SA, int ldsa,
                  magmaDoubleComplex *A,       int lda,
                  double RMAX , sycl::nd_item<3> item_ct1)
{
    int inner_bsize = item_ct1.get_local_range(2);
    int outer_bsize = inner_bsize * 512;
    int thread_id = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
            // global thread index

    if( thread_id < M ){
        for (int i =
                 outer_bsize * item_ct1.get_group(2) + item_ct1.get_local_id(2);
             i < min(M, outer_bsize * (item_ct1.get_group(2) + 1));
             i += inner_bsize) {
            A[i] = cuComplexFloatToDouble( SA[i] );
        }
    }
}

/**
    Purpose
    -------
    CLAG2Z converts a COMPLEX matrix SA to a COMPLEX_16
    matrix A.
    
    RMAX is the overflow for the COMPLEX arithmetic.
    CLAG2Z checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    ---------
    @param[in]
    M       INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    @param[in]
    N       INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    @param[in]
    SA      COMPLEX array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    @param[out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix A; if
            INFO>0, the content of A is unspecified.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" void
magmablas_clag2z_sparse(
    magma_int_t M, magma_int_t N,
    const magmaFloatComplex *SA, magma_int_t ldsa,
    magmaDoubleComplex *A,       magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info )
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    /*
    (TODO note from original dense source)
    
    Note
    ----
          - We have to provide INFO at the end that zlag2c isn't doable now.
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    */
    
    *info = 0;
    if ( M < 0 )
        *info = -1;
    else if ( N < 0 )
        *info = -2;
    else if ( lda < max(1,M) )
        *info = -4;
    else if ( ldsa < max(1,M) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
    
    double RMAX = (double)lapackf77_slamch("O");

    int block;
    sycl::range<3> dimBlock(1, 1, blksize); // Number of Threads per Block
    block = (M/blksize)/blksize;
    if (block*blksize*blksize < M)
        block++;
    sycl::range<3> dimGrid(1, 1, block); // Number of Blocks

    sycl::range<3> threads(1, 1, blksize);
    sycl::range<3> grid(1, 1, magma_ceildiv(M, blksize));
    q_ct1.memcpy(flag.get_ptr(), info, sizeof(flag)).wait(); // flag = 0
    /*
    DPCT1049:426: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream))
        ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                           magmaint_clag2z_sparse(M, N, SA, lda, A, ldsa, RMAX,
                                                  item_ct1);
                       });
    q_ct1.memcpy(info, flag.get_ptr(), sizeof(flag)).wait(); // info = flag
}
