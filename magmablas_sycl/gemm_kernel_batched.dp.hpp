#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar

       See [zcds]gemm_fermi.cu for description of related files.
*/

/******************************************************************************/
extern "C" static 
void batched_gemm_kernel_name(precision)(
    int M, int N, int K,
    FloatingPoint_t const * const * Aarray, int LDA,
    FloatingPoint_t const * const * Barray, int LDB,
    FloatingPoint_t**       Carray, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
    int batchid = blockIdx.z;
    devfunc_name(precision)( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}
