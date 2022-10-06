/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_AUXILIARY_H
#define MAGMA_AUXILIARY_H

#include "magma_types.h"

#include <math.h>  // sqrtf

#ifdef __cplusplus
extern "C" {
#endif


// =============================================================================
// initialization

magma_int_t magma_init( void );
magma_int_t magma_finalize( void );

#ifdef MAGMA_HAVE_OPENCL
magma_int_t magma_init_opencl(
    cl_platform_id platform,
    cl_context context,
    magma_int_t setup_clBlas );

magma_int_t magma_finalize_opencl(
    magma_int_t finalize_clBlas );
#endif


// =============================================================================
// version information

void magma_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro );
void magma_print_environment();


// =============================================================================
// timing

real_Double_t magma_wtime( void );
real_Double_t magma_sync_wtime( magma_queue_t queue );


// =============================================================================
// misc. functions

// magma GPU-complex PCIe connection
magma_int_t magma_buildconnection_mgpu(
    magma_int_t gnode[MagmaMaxGPUs+2][MagmaMaxGPUs+2],
    magma_int_t *ncmplx,
    magma_int_t ngpu );

void magma_indices_1D_bcyclic(
    magma_int_t nb, magma_int_t ngpu, magma_int_t dev,
    magma_int_t j0, magma_int_t j1,
    magma_int_t* dj0, magma_int_t* dj1 );

void magma_swp2pswp(
    magma_trans_t trans, magma_int_t n,
    magma_int_t *ipiv,
    magma_int_t *newipiv );


// =============================================================================
// get NB blocksize

magma_int_t magma_get_smlsize_divideconquer();


// =============================================================================
// memory allocation

magma_int_t
magma_malloc( magma_ptr *ptr_ptr, size_t bytes );

magma_int_t
magma_malloc_cpu( void **ptr_ptr, size_t bytes );

magma_int_t
magma_malloc_pinned( void **ptr_ptr, size_t bytes );

magma_int_t
magma_free_cpu( void *ptr );

#define magma_free( ptr ) \
        magma_free_internal( ptr, __func__, __FILE__, __LINE__ )

#define magma_free_pinned( ptr ) \
        magma_free_pinned_internal( ptr, __func__, __FILE__, __LINE__ )

magma_int_t
magma_free_internal(
    magma_ptr ptr,
    const char* func, const char* file, int line );

magma_int_t
magma_free_pinned_internal(
    void *ptr,
    const char* func, const char* file, int line );

// returns memory info (basically a wrapper around cudaMemGetInfo
magma_int_t
magma_mem_info(size_t* freeMem, size_t* totalMem);

// wrapper around cudaMemset
magma_int_t
magma_memset(void * ptr, int value, size_t count);

// wrapper around cudaMemsetAsync
magma_int_t
magma_memset_async(void * ptr, int value, size_t count, magma_queue_t queue);

// type-safe convenience functions to avoid using (void**) cast and sizeof(...)
// here n is the number of elements (floats, doubles, etc.) not the number of bytes.
/******************************************************************************/
/// @addtogroup magma_malloc
/// imalloc, smalloc, etc.
/// @{

/// Type-safe version of magma_malloc(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
static inline magma_int_t magma_imalloc( magmaInt_ptr           *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magma_int_t)        ); }

/// Type-safe version of magma_malloc(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
static inline magma_int_t magma_index_malloc( magmaIndex_ptr    *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magma_index_t)      ); }

/// Type-safe version of magma_malloc(), for magma_uindex_t arrays. Allocates n*sizeof(magma_uindex_t) bytes.
static inline magma_int_t magma_uindex_malloc( magmaUIndex_ptr    *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magma_uindex_t)      ); }

/// Type-safe version of magma_malloc(), for float arrays. Allocates n*sizeof(float) bytes.
static inline magma_int_t magma_smalloc( magmaFloat_ptr         *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(float)              ); }

/// Type-safe version of magma_malloc(), for double arrays. Allocates n*sizeof(double) bytes.
static inline magma_int_t magma_dmalloc( magmaDouble_ptr        *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(double)             ); }

/// Type-safe version of magma_malloc(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
static inline magma_int_t magma_cmalloc( magmaFloatComplex_ptr  *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }

/// Type-safe version of magma_malloc(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
static inline magma_int_t magma_zmalloc( magmaDoubleComplex_ptr *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

/// @}


/******************************************************************************/
/// @addtogroup magma_malloc_cpu
/// imalloc_cpu, smalloc_cpu, etc.
/// @{

/// Type-safe version of magma_malloc_cpu(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
static inline magma_int_t magma_imalloc_cpu( magma_int_t        **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magma_int_t)        ); }

/// Type-safe version of magma_malloc_cpu(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
static inline magma_int_t magma_index_malloc_cpu( magma_index_t **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magma_index_t)      ); }

/// Type-safe version of magma_malloc_cpu(), for magma_uindex_t arrays. Allocates n*sizeof(magma_uindex_t) bytes.
static inline magma_int_t magma_uindex_malloc_cpu( magma_uindex_t **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magma_uindex_t)      ); }

/// Type-safe version of magma_malloc_cpu(), for float arrays. Allocates n*sizeof(float) bytes.
static inline magma_int_t magma_smalloc_cpu( float              **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(float)              ); }

/// Type-safe version of magma_malloc_cpu(), for double arrays. Allocates n*sizeof(double) bytes.
static inline magma_int_t magma_dmalloc_cpu( double             **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(double)             ); }

/// Type-safe version of magma_malloc_cpu(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
static inline magma_int_t magma_cmalloc_cpu( magmaFloatComplex  **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }

/// Type-safe version of magma_malloc_cpu(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
static inline magma_int_t magma_zmalloc_cpu( magmaDoubleComplex **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

/// @}


/******************************************************************************/
/// @addtogroup magma_malloc_pinned
/// imalloc_pinned, smalloc_pinned, etc.
/// @{

/// Type-safe version of magma_malloc_pinned(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
static inline magma_int_t magma_imalloc_pinned( magma_int_t        **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magma_int_t)        ); }

/// Type-safe version of magma_malloc_pinned(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
static inline magma_int_t magma_index_malloc_pinned( magma_index_t **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magma_index_t)      ); }

/// Type-safe version of magma_malloc_pinned(), for float arrays. Allocates n*sizeof(float) bytes.
static inline magma_int_t magma_smalloc_pinned( float              **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(float)              ); }

/// Type-safe version of magma_malloc_pinned(), for double arrays. Allocates n*sizeof(double) bytes.
static inline magma_int_t magma_dmalloc_pinned( double             **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(double)             ); }

/// Type-safe version of magma_malloc_pinned(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
static inline magma_int_t magma_cmalloc_pinned( magmaFloatComplex  **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }

/// Type-safe version of magma_malloc_pinned(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
static inline magma_int_t magma_zmalloc_pinned( magmaDoubleComplex **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

/// @}

// CUDA MAGMA only
magma_int_t magma_is_devptr( const void* ptr );


// =============================================================================
// device support

magma_int_t
magma_num_gpus( void );
/* todo: num_accelerators */
/* todo: total accelerators? available accelerators? i.e., number to use vs. number available. */

// CUDA MAGMA only
magma_int_t
magma_getdevice_arch();
/* magma_int_t magma_getdevice_arch( magma_int_t dev or queue );   todo: new */

void
magma_getdevices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    num_dev );

void
magma_getdevice( magma_device_t* dev );

void
magma_setdevice( magma_device_t dev );

size_t
magma_mem_size( magma_queue_t queue );

magma_int_t
magma_getdevice_multiprocessor_count();

magma_int_t
magma_getdevice_num_threads_block();

magma_int_t
magma_getdevice_num_threads_multiprocessor();

size_t
magma_getdevice_shmem_block();

size_t
magma_getdevice_shmem_block_optin();

size_t
magma_getdevice_shmem_multiprocessor();

// =============================================================================
// queue support
// new magma_queue_create adds device
#define magma_queue_create(          device, queue_ptr ) \
        magma_queue_create_internal( device, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_create_from_cuda(          device, cuda_stream, cublas_handle, cusparse_handle, queue_ptr ) \
        magma_queue_create_from_cuda_internal( device, cuda_stream, cublas_handle, cusparse_handle, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_create_from_hip(           device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr ) \
        magma_queue_create_from_hip_internal( device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_create_from_sycl(          device, sycl_stream, syclblas_handle, syclsparse_handle, queue_ptr ) \
        magma_queue_create_from_sycl_internal( device, sycl_stream, syclblas_handle, syclsparse_handle, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_create_from_opencl(          device, cl_queue, queue_ptr ) \
        magma_queue_create_from_opencl_internal( device, cl_queue, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_destroy( queue ) \
        magma_queue_destroy_internal( queue, __func__, __FILE__, __LINE__ )

#define magma_queue_sync( queue ) \
        magma_queue_sync_internal( queue, __func__, __FILE__, __LINE__ )

void
magma_queue_create_internal(
    magma_device_t device,
    magma_queue_t* queue_ptr,
    const char* func, const char* file, int line );

#ifdef MAGMA_HAVE_CUDA
void
magma_queue_create_from_cuda_internal(
    magma_device_t   device,
    cudaStream_t     stream,
    cublasHandle_t   cublas,
    cusparseHandle_t cusparse,
    magma_queue_t*   queue_ptr,
    const char* func, const char* file, int line );
#endif


#ifdef MAGMA_HAVE_HIP
void
magma_queue_create_from_hip_internal(
    magma_device_t    device,
    hipStream_t       stream,
    hipblasHandle_t   hipblas,
    hipsparseHandle_t hipsparse,
    magma_queue_t*    queue_ptr,
    const char* func, const char* file, int line );
#endif


#ifdef MAGMA_HAVE_OPENCL
magma_int_t
magma_queue_create_from_opencl_internal(
    magma_device_t   device,
    cl_command_queue cl_queue,
    const char* func, const char* file, int line );
#endif

void
magma_queue_destroy_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_queue_sync_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line );

magma_int_t
magma_queue_get_device( magma_queue_t queue );


// =============================================================================
// event support

void
magma_event_create( magma_event_t* event_ptr );

void
magma_event_create_untimed( magma_event_t* event_ptr );

void
magma_event_destroy( magma_event_t event );

void
magma_event_record( magma_event_t event, magma_queue_t queue );

void
magma_event_query( magma_event_t event );

void
magma_event_sync( magma_event_t event );

void
magma_queue_wait_event( magma_queue_t queue, magma_event_t event );


// =============================================================================
// error handler

void magma_xerbla( const char *name, magma_int_t info );

const char* magma_strerror( magma_int_t error );


// =============================================================================
// string functions

size_t magma_strlcpy( char *dst, const char *src, size_t size );


// =============================================================================
// integer functions

/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
/// @ingroup magma_ceildiv
#ifndef MAGMA_HAVE_SYCL
__host__ __device__
#endif
static inline magma_int_t magma_ceildiv( magma_int_t x, magma_int_t y )
{
    return (x + y - 1)/y;
}

/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// That is, ceil(x/y)*y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
/// @ingroup magma_ceildiv
#ifndef MAGMA_HAVE_SYCL
__host__ __device__
#endif
static inline magma_int_t magma_roundup( magma_int_t x, magma_int_t y )
{
    return magma_ceildiv( x, y ) * y;
}


// =============================================================================
// scalar functions

// real and complex square root
// sqrt alone cannot be caught by the generation script because of tsqrt

/// @return Square root of x. @ingroup magma_sqrt
static inline float  magma_ssqrt( float  x ) { return sqrtf( x ); }

/// @return Square root of x. @ingroup magma_sqrt
static inline double magma_dsqrt( double x ) { return sqrt( x ); }

/// @return Complex square root of x. @ingroup magma_sqrt
magmaFloatComplex    magma_csqrt( magmaFloatComplex  x );

/// @return Complex square root of x. @ingroup magma_sqrt
magmaDoubleComplex   magma_zsqrt( magmaDoubleComplex x );


// =============================================================================
// integer print functions

void magma_iprint( magma_int_t m, magma_int_t n, const magma_int_t *A, magma_int_t lda );

void magma_iprint_gpu( magma_int_t m, magma_int_t n, magma_int_t* dA, magma_int_t ldda, magma_queue_t queue );

#ifdef __cplusplus
}
#endif


#endif // MAGMA_AUXILIARY_H
