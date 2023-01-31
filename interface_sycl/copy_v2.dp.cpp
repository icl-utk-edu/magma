/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "error.h"

#if defined(MAGMA_HAVE_SYCL)

// Generic, type-independent routines to copy data.
// Type-safe versions which avoid the user needing sizeof(...) are in headers;
// see magma_{s,d,c,z,i,index_}{set,get,copy}{matrix,vector}

/***************************************************************************//**
    @fn magma_setvector( n, elemSize, hx_src, incx, dy_dst, incy, queue )

    Copy vector hx_src on CPU host to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See magma_setvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hx_src      Source array of dimension (1 + (n-1))*incx, on CPU host.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_setvector
*******************************************************************************/
extern "C" void magma_setvector_internal(magma_int_t n, magma_int_t elemSize,
                                         void const *hx_src, magma_int_t incx,
                                         magma_ptr dy_dst, magma_int_t incy,
                                         magma_queue_t queue, const char *func,
                                         const char *file, int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    int status;
    /*
    DPCT1018:24: The cublasSetVectorAsync was migrated, but due to parameter(s)
    int(incx) and/or int(incy) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)dy_dst, (void *)hx_src, int(incy),
                                    int(incx), 1, int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    if ( queue != NULL )
        stream->wait();
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_setvector_async( n, elemSize, hx_src, incx, dy_dst, incy, queue )

    Copy vector hx_src on CPU host to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hx_src is pinned CPU memory.
    See magma_setvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hx_src      Source array of dimension (1 + (n-1))*incx, on CPU host.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_setvector
*******************************************************************************/
extern "C" void magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize, void const *hx_src, magma_int_t incx,
    magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char *func,
    const char *file, int line) try {
    // for backwards compatability, accepts NULL queue to mean NULL stream.
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    int status;
    /*
    DPCT1018:26: The cublasSetVectorAsync was migrated, but due to parameter(s)
    int(incx) and/or int(incy) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)dy_dst, (void *)hx_src, int(incy),
                                    int(incx), 1, int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_getvector( n, elemSize, dx_src, incx, hy_dst, incy, queue )

    Copy vector dx_src on GPU device to hy_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See magma_getvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    hy_dst      Destination array of dimension (1 + (n-1))*incy, on CPU host.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_getvector
*******************************************************************************/
extern "C" void magma_getvector_internal(magma_int_t n, magma_int_t elemSize,
                                         magma_const_ptr dx_src,
                                         magma_int_t incx, void *hy_dst,
                                         magma_int_t incy, magma_queue_t queue,
                                         const char *func, const char *file,
                                         int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    int status;
    /*
    DPCT1018:28: The cublasGetVectorAsync was migrated, but due to parameter(s)
    int(incx) and/or int(incy) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)hy_dst, (void *)dx_src, int(incy),
                                    int(incx), 1, int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    if ( queue != NULL )
        stream->wait();
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_getvector_async( n, elemSize, dx_src, incx, hy_dst, incy, queue )

    Copy vector dx_src on GPU device to hy_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hy_dst is pinned CPU memory.
    See magma_getvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    hy_dst      Destination array of dimension (1 + (n-1))*incy, on CPU host.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_getvector
*******************************************************************************/
extern "C" void magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src,
    magma_int_t incx, void *hy_dst, magma_int_t incy, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    // for backwards compatability, accepts NULL queue to mean NULL stream.
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    int status;
    /*
    DPCT1018:30: The cublasGetVectorAsync was migrated, but due to parameter(s)
    int(incx) and/or int(incy) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)hy_dst, (void *)dx_src, int(incy),
                                    int(incx), 1, int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_copyvector( n, elemSize, dx_src, incx, dy_dst, incy, queue )

    Copy vector dx_src on GPU device to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dx and dy can be on different GPUs.

    This version synchronizes the queue after the transfer.
    See magma_copyvector_async() for an asynchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_copyvector
*******************************************************************************/
// TODO compare performance with cublasZcopy BLAS function.
// But this implementation can handle any element size, not just [sdcz] precisions.
extern "C" void magma_copyvector_internal(magma_int_t n, magma_int_t elemSize,
                                          magma_const_ptr dx_src,
                                          magma_int_t incx, magma_ptr dy_dst,
                                          magma_int_t incy, magma_queue_t queue,
                                          const char *func, const char *file,
                                          int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    if ( incx == 1 && incy == 1 ) {
        int status;
        /*
        DPCT1003:32: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (stream->memcpy(dy_dst, dx_src, int(n * elemSize)), 0);
        if ( queue != NULL )
            stream->wait();
        check_xerror( status, func, file, line );
        MAGMA_UNUSED( status );
    }
    else {
        magma_copymatrix_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_copyvector_async( n, elemSize, dx_src, incx, dy_dst, incy, queue )

    Copy vector dx_src on GPU device to dy_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dx and dy can be on different GPUs.

    This version is asynchronous: it may return before the transfer finishes.
    See magma_copyvector() for a synchronous version.

    @param[in]
    n           Number of elements in vector.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dx_src      Source array of dimension (1 + (n-1))*incx, on GPU device.

    @param[in]
    incx        Increment between elements of hx_src. incx > 0.

    @param[out]
    dy_dst      Destination array of dimension (1 + (n-1))*incy, on GPU device.

    @param[in]
    incy        Increment between elements of dy_dst. incy > 0.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_copyvector
*******************************************************************************/
extern "C" void magma_copyvector_async_internal(
    magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src,
    magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    // for backwards compatability, accepts NULL queue to mean NULL stream.
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    if ( incx == 1 && incy == 1 ) {
        int status;
        /*
        DPCT1003:33: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (stream->memcpy(dy_dst, dx_src, int(n * elemSize)), 0);
        check_xerror( status, func, file, line );
        MAGMA_UNUSED( status );
    }
    else {
        magma_copymatrix_async_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_setmatrix( m, n, elemSize, hA_src, lda, dB_dst, lddb, queue )

    Copy all or part of matrix hA_src on CPU host to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See magma_setmatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hA_src      Source array of dimension (lda,n), on CPU host.

    @param[in]
    lda         Leading dimension of matrix A. lda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_setmatrix
*******************************************************************************/
extern "C" void magma_setmatrix_internal(magma_int_t m, magma_int_t n,
                                         magma_int_t elemSize,
                                         void const *hA_src, magma_int_t lda,
                                         magma_ptr dB_dst, magma_int_t lddb,
                                         magma_queue_t queue, const char *func,
                                         const char *file, int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        stream = &dpct::get_default_queue();
    }
    int status;
    /*
    DPCT1018:34: The cublasSetMatrixAsync was migrated, but due to parameter(s)
    int(lda) and/or int(lddb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)dB_dst, (void *)hA_src, int(lddb),
                                    int(lda), int(m), int(n), int(elemSize),
                                    dpct::automatic, *stream, false),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_setmatrix_async( m, n, elemSize, hA_src, lda, dB_dst, lddb, queue )

    Copy all or part of matrix hA_src on CPU host to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hA_src is pinned CPU memory.
    See magma_setmatrix() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    hA_src      Source array of dimension (lda,n), on CPU host.

    @param[in]
    lda         Leading dimension of matrix A. lda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_setmatrix
*******************************************************************************/
extern "C" void magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize, void const *hA_src,
    magma_int_t lda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    // for backwards compatability, accepts NULL queue to mean NULL stream.
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    int status;
    /*
    DPCT1018:36: The cublasSetMatrixAsync was migrated, but due to parameter(s)
    int(lda) and/or int(lddb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)dB_dst, (void *)hA_src, int(lddb),
                                    int(lda), int(m), int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_getmatrix( m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue )

    Copy all or part of matrix dA_src on GPU device to hB_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version synchronizes the queue after the transfer.
    See magma_getmatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    hB_dst      Destination array of dimension (ldb,n), on CPU host.

    @param[in]
    ldb         Leading dimension of matrix B. ldb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_getmatrix
*******************************************************************************/
extern "C" void
magma_getmatrix_internal(magma_int_t m, magma_int_t n, magma_int_t elemSize,
                         magma_const_ptr dA_src, magma_int_t ldda, void *hB_dst,
                         magma_int_t ldb, magma_queue_t queue, const char *func,
                         const char *file, int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        stream = &dpct::get_default_queue();
    }
    int status;
    /*
    DPCT1018:38: The cublasGetMatrixAsync was migrated, but due to parameter(s)
    int(ldda) and/or int(ldb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)hB_dst, (void *)dA_src, int(ldb),
                                    int(ldda), int(m), int(n), int(elemSize),
                                    dpct::automatic, *stream, false),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_getmatrix_async( m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue )

    Copy all or part of matrix dA_src on GPU device to hB_dst on CPU host.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.

    This version is asynchronous: it may return before the transfer finishes,
    if hB_dst is pinned CPU memory.
    See magma_getmatrix() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    hB_dst      Destination array of dimension (ldb,n), on CPU host.

    @param[in]
    ldb         Leading dimension of matrix B. ldb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_getmatrix
*******************************************************************************/
extern "C" void magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src,
    magma_int_t ldda, void *hB_dst, magma_int_t ldb, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    int status;
    /*
    DPCT1018:40: The cublasGetMatrixAsync was migrated, but due to parameter(s)
    int(ldda) and/or int(ldb) could not be evaluated, the generated code
    performance may be sub-optimal.
    */
    /*
    DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::matrix_mem_copy((void *)hB_dst, (void *)dA_src, int(ldb),
                                    int(ldda), int(m), int(n), int(elemSize),
                                    dpct::automatic, *stream, true),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_copymatrix( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue )

    Copy all or part of matrix dA_src on GPU device to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dA and dB can be on different GPUs.

    This version synchronizes the queue after the transfer.
    See magma_copymatrix_async() for an asynchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n).

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_copymatrix
*******************************************************************************/
extern "C" void magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src,
    magma_int_t ldda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        stream = &dpct::get_default_queue();
    }
    int status;
    /*
    DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::async_dpct_memcpy(dB_dst, int(lddb * elemSize), dA_src,
                                      int(ldda * elemSize), int(m * elemSize),
                                      int(n), dpct::device_to_device, *stream),
              0);
    if ( queue != NULL )
        stream->wait();
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_copymatrix_async( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue )

    Copy all or part of matrix dA_src on GPU device to dB_dst on GPU device.
    Elements may be arbitrary size.
    Type-safe versions set elemSize appropriately.
    With CUDA unified addressing, dA and dB can be on different GPUs.

    This version is asynchronous: it may return before the transfer finishes.
    See magma_copyvector() for a synchronous version.

    @param[in]
    m           Number of rows of matrix A. m >= 0.

    @param[in]
    n           Number of columns of matrix A. n >= 0.

    @param[in]
    elemSize    Size of each element, e.g., sizeof(double).

    @param[in]
    dA_src      Source array of dimension (ldda,n), on GPU device.

    @param[in]
    ldda        Leading dimension of matrix A. ldda >= m.

    @param[out]
    dB_dst      Destination array of dimension (lddb,n), on GPU device.

    @param[in]
    lddb        Leading dimension of matrix B. lddb >= m.

    @param[in]
    queue       Queue to execute in.

    @ingroup magma_copymatrix
*******************************************************************************/
extern "C" void magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src,
    magma_int_t ldda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue,
    const char *func, const char *file, int line) try {
    // for backwards compatability, accepts NULL queue to mean NULL stream.
    sycl::queue *stream = NULL;
    if ( queue != NULL ) {
        stream = queue->sycl_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    int status;
    /*
    DPCT1003:43: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    status = (dpct::async_dpct_memcpy(dB_dst, int(lddb * elemSize), dA_src,
                                      int(ldda * elemSize), int(m * elemSize),
                                      int(n), dpct::device_to_device, *stream),
              0);
    check_xerror( status, func, file, line );
    MAGMA_UNUSED( status );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif // MAGMA_HAVE_SYCL
