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
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <map>

#if __cplusplus >= 201103  // C++11 standard
#include <mutex>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(MAGMA_WITH_ACML)
#include <acml.h>
#endif

// defining MAGMA_LAPACK_H is a hack to NOT include magma_lapack.h
// via magma_internal.h here, since it conflicts with acml.h and we don't
// need lapack here, but we want acml.h for the acmlversion() function.
#define MAGMA_LAPACK_H

#include "magma_internal.h"
#include "error.h"
#include <chrono>

#define MAX_BATCHCOUNT    (65534)

#if defined(MAGMA_HAVE_SYCL)

#ifdef DEBUG_MEMORY
// defined in alloc.cpp
extern std::map< void*, size_t > g_pointers_dev;
extern std::map< void*, size_t > g_pointers_cpu;
extern std::map< void*, size_t > g_pointers_pin;
#endif

// -----------------------------------------------------------------------------
// prototypes
extern "C" void
magma_warn_leaks( const std::map< void*, size_t >& pointers, const char* type );


// -----------------------------------------------------------------------------
// constants

// bit flags
enum {
    own_none     = 0x0000,
    own_stream   = 0x0001,
    own_syclblas   = 0x0002,
    own_syclsparse = 0x0004,
    own_opencl   = 0x0008,
};


// -----------------------------------------------------------------------------
// globals
#if __cplusplus >= 201103  // C++11 standard
    static std::mutex g_mutex;
#else
    // without C++11, wrap pthread mutex
    class PthreadMutex {
    public:
        PthreadMutex()
        {
            int err = pthread_mutex_init( &mutex, NULL );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_init failed: %d\n", err );
            }
        }

        ~PthreadMutex()
        {
            int err = pthread_mutex_destroy( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_destroy failed: %d\n", err );
            }
        }

        void lock()
        {
            int err = pthread_mutex_lock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_lock failed: %d\n", err );
            }
        }

        void unlock()
        {
            int err = pthread_mutex_unlock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_unlock failed: %d\n", err );
            }
        }

    private:
        pthread_mutex_t mutex;
    };

    static PthreadMutex g_mutex;
#endif

// count of (init - finalize) calls
static int g_init = 0;

#ifndef MAGMA_NO_V1
    magma_queue_t* g_null_queues = NULL;

    #ifdef HAVE_PTHREAD_KEY
    pthread_key_t g_magma_queue_key;
    #else
    magma_queue_t g_magma_queue = NULL;
    #endif
#endif // MAGMA_NO_V1


// -----------------------------------------------------------------------------
// subset of the CUDA device properties, set by magma_init()
struct magma_device_info
{
    size_t memory;
    size_t shmem_block;      // maximum shared memory per thread block in bytes
    size_t shmem_multiproc;  // maximum shared memory per multiprocessor in bytes
    magma_int_t cuda_arch;
    magma_int_t multiproc_count;    // number of multiprocessors
};

int g_magma_devices_cnt = 0;
struct magma_device_info* g_magma_devices = NULL;


// =============================================================================
// initialization

/***************************************************************************//**
    Initializes the MAGMA library.
    Caches information about available CUDA devices.

    Every magma_init call must be paired with a magma_finalize call.
    Only one thread needs to call magma_init and magma_finalize,
    but every thread may call it. If n threads call magma_init,
    the n-th call to magma_finalize will release resources.

    When renumbering CUDA devices, call cudaSetValidDevices before calling magma_init.
    When setting CUDA device flags, call cudaSetDeviceFlags before calling magma_init.

    @retval MAGMA_SUCCESS
    @retval MAGMA_ERR_UNKNOWN
    @retval MAGMA_ERR_HOST_ALLOC

    @see magma_finalize

    @ingroup magma_init
*******************************************************************************/
extern "C" magma_int_t magma_init() try {
    magma_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init == 0 ) {
            // query number of devices
            int err;
            g_magma_devices_cnt = 0;
            /*
            DPCT1003:45: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            err =
                (g_magma_devices_cnt = dpct::dev_mgr::instance().device_count(),
                 0);
            if (err != 0 && err != 100) {
                info = MAGMA_ERR_UNKNOWN;
                goto cleanup;
            }

            // allocate list of devices
            size_t size;
            size = max( 1, g_magma_devices_cnt ) * sizeof(struct magma_device_info);
            magma_malloc_cpu( (void**) &g_magma_devices, size );
            if ( g_magma_devices == NULL ) {
                info = MAGMA_ERR_HOST_ALLOC;
                goto cleanup;
            }
            memset( g_magma_devices, 0, size );

            // query each device
            for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
               if (!(dpct::dev_mgr::instance().get_device(dev).is_host())) {
                dpct::device_info prop;
                /*
                DPCT1003:46: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                err =
                    (dpct::dev_mgr::instance().get_device(dev).get_device_info(
                         prop),
                     0);
                if ( err != 0 ) {
                    info = MAGMA_ERR_UNKNOWN;
                }
                else {
                    g_magma_devices[dev].memory = prop.get_global_mem_size();
                    /*
                    DPCT1019:47: local_mem_size in SYCL is not a complete
                    equivalent of sharedMemPerBlock in CUDA. You may need to
                    adjust the code.
                    */
                    g_magma_devices[dev].shmem_block =
                        prop.get_local_mem_size();
                    
		    // TODO:sharedMemPerMultiprocessor not part of dpct prop
                    #ifdef MAGMA_HAVE_CUDA
                    /*
                    DPCT1005:48: The SYCL device version is different from CUDA
                    Compute Compatibility. You may need to rewrite this code.
                    */
                    g_magma_devices[dev].cuda_arch =
                        prop.get_major_version() * 100 +
                        prop.get_minor_version() * 10;
                    g_magma_devices[dev].shmem_multiproc = prop.sharedMemPerMultiprocessor;
                    #elif defined(MAGMA_HAVE_HIP)
                    g_magma_devices[dev].cuda_arch       = prop.gcnArch;
                    g_magma_devices[dev].shmem_multiproc = prop.maxSharedMemoryPerMultiProcessor;
                    #endif

                    g_magma_devices[dev].multiproc_count =
                        prop.get_max_compute_units();
                }
              }
	    }

            #ifndef MAGMA_NO_V1
                #ifdef HAVE_PTHREAD_KEY
                    // create thread-specific key
                    // currently, this is needed only for MAGMA v1 compatability
                    // see magma_init, magmablas(Set|Get)KernelStream, magmaGetQueue
                    info = pthread_key_create( &g_magma_queue_key, NULL );
                    if ( info != 0 ) {
                        info = MAGMA_ERR_UNKNOWN;
                        goto cleanup;
                    }
                #endif

                // ----- queues with NULL streams (for backwards compatability with MAGMA 1.x)
                // allocate array of queues with NULL stream
                size = max( 1, g_magma_devices_cnt ) * sizeof(magma_queue_t);
                magma_malloc_cpu( (void**) &g_null_queues, size );
                if ( g_null_queues == NULL ) {
                    info = MAGMA_ERR_HOST_ALLOC;
                    goto cleanup;
                }
                memset( g_null_queues, 0, size );
            #endif // MAGMA_NO_V1
        }
cleanup:
        g_init += 1;  // increment (init - finalize) count
    }
    g_mutex.unlock();

    return info;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Frees information used by the MAGMA library.
    @see magma_init

    @ingroup magma_init
*******************************************************************************/
extern "C" magma_int_t
magma_finalize()
{
    magma_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init <= 0 ) {
            info = MAGMA_ERR_NOT_INITIALIZED;
        }
        else {
            g_init -= 1;  // decrement (init - finalize) count
            if ( g_init == 0 ) {
                info = 0;

                if ( g_magma_devices != NULL ) {
                    magma_free_cpu( g_magma_devices );
                    g_magma_devices = NULL;
                }

                #ifndef MAGMA_NO_V1
                if ( g_null_queues != NULL ) {
                    for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
                        magma_queue_destroy( g_null_queues[dev] );
                        g_null_queues[dev] = NULL;
                    }
                    magma_free_cpu( g_null_queues );
                    g_null_queues = NULL;
                }

                #ifdef HAVE_PTHREAD_KEY
                    pthread_key_delete( g_magma_queue_key );
                #endif
                #endif // MAGMA_NO_V1

                #ifdef DEBUG_MEMORY
                magma_warn_leaks( g_pointers_dev, "device" );
                magma_warn_leaks( g_pointers_cpu, "CPU" );
                magma_warn_leaks( g_pointers_pin, "CPU pinned" );
                #endif
            }
        }
    }
    g_mutex.unlock();

    return info;
}


// =============================================================================
// testing and debugging support

#ifdef DEBUG_MEMORY
/***************************************************************************//**
    If DEBUG_MEMORY is defined at compile time, prints warnings when
    magma_finalize() is called for any GPU device, CPU, or CPU pinned
    allocations that were not freed.

    @param[in]
    pointers    Hash table mapping allocated pointers to size.

    @param[in]
    type        String describing type of pointers (GPU, CPU, etc.)

    @ingroup magma_testing
*******************************************************************************/
extern "C" void
magma_warn_leaks( const std::map< void*, size_t >& pointers, const char* type )
{
    if ( pointers.size() > 0 ) {
        fprintf( stderr, "Warning: MAGMA detected memory leak of %llu %s pointers:\n",
                 (long long unsigned) pointers.size(), type );
        std::map< void*, size_t >::const_iterator iter;
        for( iter = pointers.begin(); iter != pointers.end(); ++iter ) {
            fprintf( stderr, "    pointer %p, size %lu\n", iter->first, iter->second );
        }
    }
}
#endif


/***************************************************************************//**
    Print MAGMA version, CUDA version, LAPACK/BLAS library version,
    available GPU devices, number of threads, date, etc.
    Used in testing.
    @ingroup magma_testing
*******************************************************************************/
extern "C" void magma_print_environment() try {
    magma_int_t major, minor, micro;
    magma_version( &major, &minor, &micro );

    printf( "%% MAGMA %lld.%lld.%lld %s %lld-bit magma_int_t, %lld-bit pointer.\n",
            (long long) major, (long long) minor, (long long) micro,
            MAGMA_VERSION_STAGE,
            (long long) (8*sizeof(magma_int_t)),
            (long long) (8*sizeof(void*)) );

/* SYCL */

#if defined(MAGMA_HAVE_SYCL)
//    printf("%% Compiled with SYCL support for %.1f\n", MAGMA_CUDA_ARCH_MIN/100.);

    // SYCL, OpenCL, OpenMP, MKL, ACML versions all printed on same line
    std::string sycl_runtime, sycl_driver;
    int err;
    /*
    DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1043:50: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    err =
        (sycl_driver =
             dpct::get_current_device().get_info<sycl::info::device::version>(),
         0);
    check_error( err );
    /*
    DPCT1003:51: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1043:52: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    err =
        (sycl_runtime =
             dpct::get_current_device().get_info<sycl::info::device::version>(),
         0);
    if (err != 100) {
        check_error( err );
    }
    printf( "%% SYCL runtime %s, driver %s. ", sycl_runtime.c_str(), sycl_driver.c_str() );

#endif

/* OpenMP */

#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "OpenMP threads %d. ", omp_threads );
#else
    printf( "MAGMA not compiled with OpenMP. " );
#endif

#if defined(MAGMA_WITH_MKL)
    MKLVersion mkl_version;
    mkl_get_version( &mkl_version );
    printf( "MKL %d.%d.%d, MKL threads %d. ",
            mkl_version.MajorVersion,
            mkl_version.MinorVersion,
            mkl_version.UpdateVersion,
            mkl_get_max_threads() );
#endif

#if defined(MAGMA_WITH_ACML)
    // ACML 4 doesn't have acml_build parameter
    int acml_major, acml_minor, acml_patch, acml_build;
    acmlversion( &acml_major, &acml_minor, &acml_patch, &acml_build );
    printf( "ACML %d.%d.%d.%d ", acml_major, acml_minor, acml_patch, acml_build );
#endif

    printf( "\n" );

    // print devices
    int ndevices = 0;
    /*
    DPCT1003:53: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (ndevices = dpct::dev_mgr::instance().device_count(), 0);
    if (err != 100) {
        check_error( err );
    }
    for( int dev = 0; dev < ndevices; ++dev ) {
        if (!(dpct::dev_mgr::instance().get_device(dev).is_host())) {
        dpct::device_info prop;
        /*
        DPCT1003:54: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        err = (dpct::dev_mgr::instance().get_device(dev).get_device_info(prop),
               0);
        check_error( err );

        #ifdef MAGMA_HAVE_SYCL
        printf("%% device %d: %s, %.1f MHz clock, %.1f MiB memory, capability "
               "%d.%d\n",
               dev, prop.get_name(), prop.get_max_clock_frequency() / 1000.,
               prop.get_global_mem_size() / (1024. * 1024.),
               /*
               DPCT1005:55: The SYCL device version is different from CUDA
               Compute Compatibility. You may need to rewrite this code.
               */
               prop.get_major_version(),
               /*
               DPCT1005:56: The SYCL device version is different from CUDA
               Compute Compatibility. You may need to rewrite this code.
               */
               prop.get_minor_version());

        /*
        DPCT1005:57: The SYCL device version is different from CUDA Compute
        Compatibility. You may need to rewrite this code.
        */
	//TODO
/*        int arch =
            prop.get_major_version() * 100 + prop.get_minor_version() * 10; */
/*        if ( arch < MAGMA_CUDA_ARCH_MIN ) {
            printf("\n"
                   "==============================================================================\n"
                   "WARNING: MAGMA was compiled only for CUDA capability %.1f and higher;\n"
                   "device %d has only capability %.1f; some routines will not run correctly!\n"
                   "==============================================================================\n\n",
                   MAGMA_CUDA_ARCH_MIN/100., dev, arch/100. );
        } */
        #endif
      }
    }
    MAGMA_UNUSED( err );
    time_t t = time( NULL );
    printf( "%% %s", ctime( &t ));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#if CUDA_VERSION >= 11000
#define magma_memoryType() type
#else
#define magma_memoryType() memoryType
#endif

/***************************************************************************//**
    For debugging purposes, determines whether a pointer points to CPU or GPU memory.

    On CUDA architecture 2.0 cards with unified addressing, CUDA can tell if
    it is a device pointer or pinned host pointer.
    For malloc'd host pointers, cudaPointerGetAttributes returns error,
    implying it is a (non-pinned) host pointer.

    On older cards, this cannot determine if it is CPU or GPU memory.

    @param[in] A    pointer to test

    @return  1:  if A is a device pointer (definitely),
    @return  0:  if A is a host   pointer (definitely or inferred from error),
    @return -1:  if unknown.

    @ingroup magma_util
*******************************************************************************/
extern "C" magma_int_t magma_is_devptr(const void *A) try {
    int err;
    dpct::device_info prop;
//    cudaPointerAttributes attr;
//    int dev;  // must be int
//    err = dev = dpct::dev_mgr::instance().current_device_id();
//    if ( ! err ) {
//        /*
//        DPCT1003:59: Migrated API does not return error code. (*, 0) is
//        inserted. You may need to rewrite this code.
//        */
//        err = (dpct::dev_mgr::instance().get_device(dev).get_device_info(prop),
//               0);
////TODO: how to handle this with SYCL?
//#ifdef MAGMA_HAVE_CUDA
//        if ( ! err && prop.unifiedAddressing ) {
//        #elif defined(MAGMA_HAVE_HIP)
//        // in HIP, assume all can.
//        /There's no corresponding property, and examples show no need to check any properties
//        if ( ! err ) {
//        #endif
//
//            // I think the cudaPointerGetAttributes prototype is wrong, missing const (mgates)
//            /*
//            DPCT1007:60: Migration of cudaPointerGetAttributes is not supported
//            by the Intel(R) DPC++ Compatibility Tool.
//            */
//  /*          err = cudaPointerGetAttributes(&attr, const_cast<void *>(A));
//            if ( ! err ) {
//                // definitely know type
//                #ifdef MAGMA_HAVE_CUDA
//                  #if CUDA_VERSION >= 11000
//                    return (attr.type == cudaMemoryTypeDevice);
//                  #else
//                    return (attr.memoryType == cudaMemoryTypeDevice);
//                  #endif
//
//                #elif defined(MAGMA_HAVE_HIP)
//                return (attr.memoryType == hipMemoryTypeDevice);
//                #endif
//            } */
//            /*
//            DPCT1002:62: Special case error handling if-stmt was detected. You
//            may need to rewrite this code.
//            */
//  //          else if (err == 1) {
//                // clear error; see http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=529
//                /*
//                DPCT1001:61: The statement could not be removed.
//                */
//                /*
//                DPCT1026:63: The call to cudaGetLastError was removed because
//                the function call is redundant in DPC++.
//                */
//    //            // infer as host pointer
//   //             return 0;
//    //        }
//    //    }
//    //}
//    // clear error
//    /*
//    DPCT1026:58: The call to cudaGetLastError was removed because the function
//    call is redundant in DPC++.
//    */
//    // unknown, e.g., device doesn't support unified addressing
    return -1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// =============================================================================
// device support

/***************************************************************************//**
    Returns CUDA architecture capability for the current device.
    This requires magma_init() to be called first to cache the information.
    Version is an integer xyz, where x is major, y is minor, and z is micro,
    the same as __CUDA_ARCH__. Thus for architecture 1.3.0 it returns 130.

    @return CUDA_ARCH for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" magma_int_t magma_getdevice_arch() try {
    int dev;
    int err;
    err = dev = dpct::dev_mgr::instance().current_device_id();
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Fills in devices array with the available devices.
    (This makes much more sense in OpenCL than in CUDA.)

    @param[out]
    devices     Array of dimension (size).
                On output, devices[0, ..., num_dev-1] contain device IDs.
                Entries >= num_dev are not touched.

    @param[in]
    size        Dimension of the array devices.

    @param[out]
    num_dev     Number of devices, limited to size.

    @ingroup magma_device
*******************************************************************************/
extern "C" void magma_getdevices(magma_device_t *devices, magma_int_t size,
                                 magma_int_t *num_dev) try {
    int err;
    int cnt;
    /*
    DPCT1003:64: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (cnt = dpct::dev_mgr::instance().device_count(), 0);
    check_error( err );
    MAGMA_UNUSED( err );

    cnt = min( cnt, int(size) );
    for( int i = 0; i < cnt; ++i ) {
        devices[i] = i;
    }
    *num_dev = cnt;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Get the current device.

    @param[out]
    device      On output, device ID of the current device.
                Each thread has its own current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" void magma_getdevice(magma_device_t *device) try {
    int dev;
    int err;
    err = dev = dpct::dev_mgr::instance().current_device_id();
    *device = dev;
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Set the current device.

    @param[in]
    device      Device ID to set as the current device.
                Each thread has its own current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" void magma_setdevice(magma_device_t device) try {
    int err;
    /*
    DPCT1093:65: The "int(device)" may not be the best XPU device. Adjust the
    selected device if needed.
    */
    /*
    DPCT1003:66: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (dpct::dev_mgr::instance().select_device(int(device)), 0);
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Returns the multiprocessor count for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the multiprocessor count for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" magma_int_t magma_getdevice_multiprocessor_count() try {
    int dev;
    int err;
    err = dev = dpct::dev_mgr::instance().current_device_id();
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].multiproc_count;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Returns the maximum shared memory per block (in bytes) for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the maximum shared memory per block (in bytes) for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" size_t magma_getdevice_shmem_block() try {
    int dev;
    int err;
    err = dev = dpct::dev_mgr::instance().current_device_id();
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].shmem_block;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Returns the maximum shared memory multiprocessor (in bytes) for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the maximum shared memory per multiprocessor (in bytes) for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" size_t magma_getdevice_shmem_multiprocessor() try {
    int dev;
    int err;
    err = dev = dpct::dev_mgr::instance().current_device_id();
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].shmem_multiproc;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @param[in]
    queue           Queue to query.

    @return         Amount of free memory in bytes available on the device
                    associated with the queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C" size_t magma_mem_size(magma_queue_t queue) try {
    // CUDA would only need a device ID, but OpenCL requires a queue.
    size_t freeMem, totalMem;
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_setdevice( magma_queue_get_device( queue ));
    /*
    DPCT1003:67: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1072:68: DPC++ currently does not support getting the available memory
    on the current device. You may need to adjust the code.
    */
    int err =
        (totalMem =
             dpct::get_current_device().get_device_info().get_global_mem_size(),
         0);
    check_error( err );
    MAGMA_UNUSED( err );
    magma_setdevice( orig_dev );
    return freeMem;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// =============================================================================
// queue support

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return Device ID associated with the MAGMA queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
magma_int_t
magma_queue_get_device( magma_queue_t queue )
{
    return queue->device();
}


#ifdef MAGMA_HAVE_SYCL
/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return SYCL stream associated with the MAGMA queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C" sycl::queue *magma_queue_get_sycl_stream(magma_queue_t queue)
{
    return queue->sycl_stream();
}


/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return syclBLAS handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/

extern "C" sycl::queue *magma_queue_get_syclblas_handle(magma_queue_t queue)
{
    return queue->syclblas_handle();
}

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return syclSparse handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/
extern "C" sycl::queue *magma_queue_get_syclsparse_handle(magma_queue_t queue)
{
    return queue->syclsparse_handle();
}

#endif



/***************************************************************************//**
    @fn magma_queue_create( device, queue_ptr )

    magma_queue_create( device, queue_ptr ) is the preferred alias to this
    function.

    Creates a new MAGMA queue, with associated CUDA stream, cuBLAS handle,
    and cuSparse handle.

    This is the MAGMA v2 version which takes a device ID.

    @param[in]
    device          Device to create queue on.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup magma_queue
*******************************************************************************/
// This was previously marked extern "C", but dpcpp was giving errors
// TODO...?
void magma_queue_create_internal(magma_device_t device,
                                            magma_queue_t *queue_ptr,
                                            const char *func, const char *file,
                                            int line) try {
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;

#if defined(MAGMA_HAVE_SYCL)
    queue->syclblas__   = NULL;
    queue->syclsparse__ = NULL;
#endif
    queue->maxbatch__ = MAX_BATCHCOUNT;

    magma_setdevice( device );

    int err;
    /*
    DPCT1003:69: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (queue->stream__ = dpct::get_current_device().create_queue(), 0);
    check_xerror( err, func, file, line );
    queue->own__ |= own_stream;

#if defined(MAGMA_HAVE_SYCL)
    int stat;
    /*
    DPCT1003:70: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat = (queue->syclblas__ = &dpct::get_default_queue(), 0);
    check_xerror( stat, func, file, line );
    queue->own__ |= own_syclblas;
    /*
    DPCT1003:71: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat = (queue->syclblas__ = queue->stream__, 0);
    check_xerror( stat, func, file, line );

    int stat2;
    /*
    DPCT1003:72: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat2 = (queue->syclsparse__ = &dpct::get_default_queue(), 0);
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_syclsparse;
    /*
    DPCT1003:73: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat2 = (queue->syclsparse__ = queue->stream__, 0);
    check_xerror( stat2, func, file, line );
#endif

    MAGMA_UNUSED( err );
    MAGMA_UNUSED( stat );
    MAGMA_UNUSED( stat2 );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_queue_create_from_sycl( device, sycl_queue, syclblas_handle, syclsparse_handle, queue_ptr )

    Warning: non-portable outside of SYCL. Use with discretion.

    Creates a new MAGMA queue, using the given SYCL queue.  TODO: blas, sparse handles?
    The caller retains ownership of the given stream and
    handles, so must free them after destroying the queue;
    see magma_queue_destroy().

    MAGMA sets the stream on the BLAS and Sparse handles, and assumes
    it will not be changed while MAGMA is running.

    @param[in]
    device          Device to create queue on.

    @param[in]
    sycl_queue      SYCL queue to use.

    @param[in]
    syclblas_handle   TODO

    @param[in]
    syclsparse_handle TODO 

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup magma_queue
*******************************************************************************/
#ifdef MAGMA_HAVE_SYCL
// This was previously marked extern "C", but dpcpp was giving errors
// TODO...?
void magma_queue_create_from_sycl_internal(
    magma_device_t device, sycl::queue *sycl_queue, sycl::queue *syclblas_handle,
    sycl::queue *syclsparse_handle, magma_queue_t *queue_ptr, const char *func,
    const char *file, int line) try {
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__        = own_none;
    queue->device__     = device;
    queue->stream__     = NULL;
    queue->syclblas__   = NULL;
    queue->syclsparse__ = NULL;
    queue->ptrArray__   = NULL;
    queue->dAarray__    = NULL;
    queue->dBarray__    = NULL;
    queue->dCarray__    = NULL;
    queue->maxbatch__   = MAX_BATCHCOUNT;

    magma_setdevice( device );

    // stream can be NULL
    queue->stream__ = sycl_queue;

    // allocate cublas handle if given as NULL
    int stat;
    if ( syclblas_handle == NULL ) {
        /*
        DPCT1003:74: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        stat = (syclblas_handle = &dpct::get_default_queue(), 0);
        check_xerror( stat, func, file, line );
        queue->own__ |= own_syclblas;
    }
    queue->syclblas__ = syclblas_handle;
    /*
    DPCT1003:75: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat = (queue->syclblas__ = queue->stream__, 0);
    check_xerror( stat, func, file, line );

    // allocate syclsparse handle if given as NULL
    int stat2;
    if ( syclsparse_handle == NULL ) {
        /*
        DPCT1003:76: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        stat2 = (syclsparse_handle = &dpct::get_default_queue(), 0);
        check_xerror( stat, func, file, line );
        queue->own__ |= own_syclsparse;
    }
    queue->syclsparse__ = syclsparse_handle;
    /*
    DPCT1003:77: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    stat2 = (queue->syclsparse__ = queue->stream__, 0);
    check_xerror( stat2, func, file, line );

    MAGMA_UNUSED( stat );
    MAGMA_UNUSED( stat2 );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif


/***************************************************************************//**
    @fn magma_queue_destroy( queue )

    Destroys a queue, freeing its resources.

    If the queue was created with magma_queue_create_from_cuda(), the CUDA
    stream, cuBLAS handle, and cuSparse handle given there are NOT freed -- the
    caller retains ownership. However, if MAGMA allocated the handles, MAGMA
    will free them here.

    @param[in]
    queue           Queue to destroy.

    @ingroup magma_queue
*******************************************************************************/
extern "C" void magma_queue_destroy_internal(magma_queue_t queue,
                                             const char *func, const char *file,
                                             int line) try {
    if ( queue != NULL ) {
        if ( queue->syclblas__ != NULL && (queue->own__ & own_syclblas)) {
            /*
            DPCT1003:78: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            int stat = (queue->syclblas__ = nullptr, 0);
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
        if ( queue->syclsparse__ != NULL && (queue->own__ & own_syclsparse)) {
            /*
            DPCT1003:79: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            int stat = (queue->syclsparse__ = nullptr, 0);
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
        if ( queue->stream__ != NULL && (queue->own__ & own_stream)) {
            /*
            DPCT1003:80: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            int err =
                (dpct::get_current_device().destroy_queue(queue->stream__), 0);
            check_xerror( err, func, file, line );
            MAGMA_UNUSED( err );
        }

        if( queue->ptrArray__ != NULL ) magma_free( queue->ptrArray__ );

        queue->own__      = own_none;
        queue->device__   = -1;
        queue->stream__   = NULL;
        queue->ptrArray__ = NULL;
        queue->dAarray__  = NULL;
        queue->dBarray__  = NULL;
        queue->dCarray__  = NULL;
        queue->syclblas__   = NULL;
        queue->syclsparse__ = NULL;
        magma_free_cpu( queue );
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    @fn magma_queue_sync( queue )

    Synchronizes with a queue. The CPU blocks until all operations on the queue
    are finished.

    @param[in]
    queue           Queue to synchronize.

    @ingroup magma_queue
*******************************************************************************/
extern "C" void magma_queue_sync_internal(magma_queue_t queue, const char *func,
                                          const char *file, int line) try {
    int err;
//    if ( queue != NULL ) {
        /*
        DPCT1003:81: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        err = (queue->sycl_stream()->wait(), 0);
//    }
//    else {
        /*
        DPCT1003:82: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
//        err = (NULL->wait(), 0);
//    }
    check_xerror( err, func, file, line );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// =============================================================================
// event support

/***************************************************************************//**
    Creates a GPU event.

    @param[in]
    event           On output, the newly created event.

    @ingroup magma_event
*******************************************************************************/
extern "C" void magma_event_create(magma_event_t *event) try {
    int err;
    /*
    DPCT1027:83: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    err = 0;
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Creates a GPU event, without timing support. May improve performance

    @param[in]
    event           On output, the newly created event.

    @ingroup magma_event
*******************************************************************************/
extern "C" void magma_event_create_untimed(magma_event_t *event) try {
    int err;
    /*
    DPCT1027:84: The call to cudaEventCreateWithFlags was replaced with 0
    because this call is redundant in DPC++.
    */
    err = 0;
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//*
    Destroys a GPU event, freeing its resources.

    @param[in]
    event           Event to destroy.

    @ingroup magma_event
*******************************************************************************/
//TODO: can we disregard this entirely for SYCL?  Considering DPCT message...
extern "C" void magma_event_destroy(magma_event_t event) try {
   // if ( event != NULL ) {
        int err;
        /*
        DPCT1027:85: The call to cudaEventDestroy was replaced with 0 because
        this call is redundant in DPC++.
        */
        err = 0;
        check_error( err );
        MAGMA_UNUSED( err );
   // }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Records an event into the queue's execution stream.
    The event will trigger when all previous operations on this queue finish.

    @param[in]
    event           Event to record.

    @param[in]
    queue           Queue to execute in.

    @ingroup magma_event
*******************************************************************************/
extern "C" void magma_event_record(magma_event_t event,
                                   magma_queue_t queue) try {
    int err;
std::chrono::time_point<std::chrono::steady_clock> event_ct1;
    /*
    DPCT1012:86: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:87: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    event_ct1 = std::chrono::steady_clock::now();
    err = (event = queue->sycl_stream()->ext_oneapi_submit_barrier(), 0);
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Synchronizes with an event. The CPU blocks until the event triggers.

    @param[in]
    event           Event to synchronize with.

    @ingroup magma_event
*******************************************************************************/
extern "C" void magma_event_sync(magma_event_t event) try {
    int err;
    /*
    DPCT1003:90: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (event.wait_and_throw(), 0);
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Synchronizes a queue with an event. The queue blocks until the event
    triggers. The CPU does not block.

    @param[in]
    event           Event to synchronize with.

    @param[in]
    queue           Queue to synchronize.

    @ingroup magma_event
*******************************************************************************/
extern "C" void magma_queue_wait_event(magma_queue_t queue,
                                       magma_event_t event) try {
    int err;
    /*
    DPCT1003:88: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err = (event = queue->sycl_stream()->ext_oneapi_submit_barrier({event}), 0);
    check_error( err );
    MAGMA_UNUSED( err );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif // MAGMA_HAVE_SYCL
