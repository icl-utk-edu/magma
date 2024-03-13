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

       See [zcds]gemm_fermi.cu for description of related files.
*/

/******************************************************************************/

extern "C" {
    
static 
void gemm_kernel_name(precision)(
    int M, int N, int K,
    const FloatingPoint_t* __restrict__ A, int LDA,
    const FloatingPoint_t* __restrict__ B, int LDB,
    FloatingPoint_t*       __restrict__ C, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB , sycl::nd_item<3> item_ct1,
    sycl::local_accessor<FloatingPoint_t, 2> sA,
    sycl::local_accessor<FloatingPoint_t, 2> sB)
{
    devfunc_name(precision)(M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta,
                            offsetA, offsetB, item_ct1, sA, sB);
}

}
