#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar

       See [zcds]zherk_fermi_batched.cu for description of related files.
*/

/******************************************************************************/
extern "C" static  
void batched_herk_kernel_name(precision)(
    magma_uplo_t uplo, int N, int K,
    FloatingPoint_t const * const * Aarray, int LDA,
    FloatingPoint_t const * const * Barray, int LDB,
    FloatingPoint_t** Carray, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB )
{
    if (blockDim.x != blockDim.y) {
        //printf("error zherk_fermi_kernel blkx=%d != blky=%d not supported where n=%d\n",blockDim.x,blockDim.y,N);
        return;
    }

    // for lower blkx > blky do not have to compute
    if ( uplo == MagmaLower && blockIdx.y > blockIdx.x )
        return;
    
    // for upper blkx < blky do not have to compute
    if ( uplo == MagmaUpper && blockIdx.y < blockIdx.x )
        return;

    int batchid = blockIdx.z;
    
    devfunc_name(precision)( N, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}
