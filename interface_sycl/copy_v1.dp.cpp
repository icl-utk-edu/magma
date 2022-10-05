/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#ifndef MAGMA_NO_V1

#include <CL/sycl.hpp>
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
                                            int line) try {
    int status;
    /*
    DPCT1018:19: The cublasSetMatrix was migrated, but due to parameter(s)
    int(lda) and/or int(lddb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)dB_dst, (void *)hA_src, int(lddb),
                                    int(lda), int(m), int(n), int(elemSize)),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/******************************************************************************/
extern "C" void magma_getmatrix_v1_internal(magma_int_t m, magma_int_t n,
                                            magma_int_t elemSize,
                                            magma_const_ptr dA_src,
                                            magma_int_t ldda, void *hB_dst,
                                            magma_int_t ldb, const char *func,
                                            const char *file, int line) try {
    int status;
    /*
    DPCT1018:21: The cublasGetMatrix was migrated, but due to parameter(s)
    int(ldda) and/or int(ldb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)hB_dst, (void *)dA_src, int(ldb),
                                    int(ldda), int(m), int(n), int(elemSize)),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/******************************************************************************/
extern "C" void magma_copymatrix_v1_internal(magma_int_t m, magma_int_t n,
                                             magma_int_t elemSize,
                                             magma_const_ptr dA_src,
                                             magma_int_t ldda, magma_ptr dB_dst,
                                             magma_int_t lddb, const char *func,
                                             const char *file, int line) try {
    int status;
    /*
    DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::dpct_memcpy(dB_dst, int(lddb * elemSize), dA_src,
                                int(ldda * elemSize), int(m * elemSize), int(n),
                                dpct::device_to_device),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif // MAGMA_HAVE_CUDA

#endif // MAGMA_NO_V1
