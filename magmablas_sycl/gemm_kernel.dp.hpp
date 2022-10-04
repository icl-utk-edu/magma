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

extern "C" {
    
static 
void gemm_kernel_name(precision)(
    int M, int N, int K,
    const FloatingPoint_t* __restrict__ A, int LDA,
    const FloatingPoint_t* __restrict__ B, int LDB,
    FloatingPoint_t*       __restrict__ C, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB , sycl::nd_item<3> item_ct1,
    sycl::accessor<FloatingPoint_t, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<FloatingPoint_t, 2, sycl::access_mode::read_write, sycl::access::target::local> sB,
    dpct::image_accessor_ext<sycl::int4, 1> tex_ref_Amagma_z,
    dpct::image_accessor_ext<sycl::int4, 1> tex_ref_Bmagma_z,
    dpct::image_accessor_ext<sycl::float2, 1> tex_ref_Amagma_c,
    dpct::image_accessor_ext<sycl::float2, 1> tex_ref_Bmagma_c,
    dpct::image_accessor_ext<sycl::int2, 1> tex_ref_Amagma_d,
    dpct::image_accessor_ext<sycl::int2, 1> tex_ref_Bmagma_d,
    dpct::image_accessor_ext<float, 1> tex_ref_Amagma_s,
    dpct::image_accessor_ext<float, 1> tex_ref_Bmagma_s)
{
    devfunc_name(precision)(
        M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, offsetA, offsetB,
        item_ct1, sA, sB, tex_ref_Amagma_z, tex_ref_Bmagma_z, tex_ref_Amagma_c,
        tex_ref_Bmagma_c, tex_ref_Amagma_d, tex_ref_Bmagma_d, tex_ref_Amagma_s,
        tex_ref_Bmagma_s);
}

}
