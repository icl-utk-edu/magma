/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#ifndef MAGMA_NO_V1

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#if defined(MAGMA_HAVE_SYCL)

// These MAGMA v1 routines are all deprecated.
// See copy_v2.cpp for documentation.

// Generic, type-independent routines to copy data.
// Type-safe versions which avoid the user needing sizeof(...) are in headers;
// see magma_{s,d,c,z,i,index_}{set,get,copy}{matrix,vector}

/******************************************************************************/
extern "C" void
magma_setvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_internal(
        n, elemSize,
        hx_src, incx,
        dy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void
magma_getvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void*           hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_internal(
        n, elemSize,
        dx_src, incx,
        hy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void
magma_copyvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal(
        n, elemSize,
        dx_src, incx,
        dy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}


/******************************************************************************/
extern "C" void magma_setmatrix_v1_internal(magma_int_t m, magma_int_t n,
                                            magma_int_t elemSize,
                                            void const *hA_src, magma_int_t lda,
                                            magma_ptr dB_dst, magma_int_t lddb,
                                            const char *func, const char *file,
                                            int line)
{
   try {
      sycl::queue *stream = &dpct::get_default_queue();
      stream->ext_oneapi_memcpy2d(
                    dB_dst, size_t(lddb) * elemSize,
		    hA_src, size_t(lda) * elemSize,
		    size_t(m) * elemSize, size_t(n));
      stream->wait();
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
}

/******************************************************************************/
extern "C" void magma_getmatrix_v1_internal(magma_int_t m, magma_int_t n,
                                            magma_int_t elemSize,
                                            magma_const_ptr dA_src,
                                            magma_int_t ldda, void *hB_dst,
                                            magma_int_t ldb, const char *func,
                                            const char *file, int line)
{
    try {
      sycl::queue *stream = &dpct::get_default_queue();
      stream->ext_oneapi_memcpy2d(
		  hB_dst, size_t(ldb) * elemSize,
		  dA_src, size_t(ldda) * elemSize,
		  size_t(m) * elemSize, size_t(n));
      stream->wait();
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
}

/******************************************************************************/
extern "C" void magma_copymatrix_v1_internal(magma_int_t m, magma_int_t n,
                                             magma_int_t elemSize,
                                             magma_const_ptr dA_src,
                                             magma_int_t ldda, magma_ptr dB_dst,
                                             magma_int_t lddb, const char *func,
                                             const char *file, int line)
{
    try {
      sycl::queue *stream = &dpct::get_default_queue();
      stream->ext_oneapi_memcpy2d(
		  dB_dst, size_t(lddb) * elemSize,
		  dA_src, size_t(ldda) * elemSize,
		  size_t(m) * elemSize, size_t(n));
      stream->wait();
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
}

#endif // MAGMA_HAVE_SYCL

#endif // MAGMA_NO_V1
